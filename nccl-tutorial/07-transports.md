# 07｜Transports：把逻辑邻居变成可用连接

上一章已经得到 `ring.prev/next` 和 `tree.up/down`。本章回答：**这些逻辑关系怎样变成 GPU 或 proxy 真正能用的缓冲、映射和连接？**

本章只讲通用交接，不逐字段展开。P2P 的 IPC 细节、SHM 内部布局和 NET 状态机见 [Transport 进阶附录](appendix/transports.md)。

## 1. 先分清两种连接对象

| 对象 | 主要作用 |
|---|---|
| `ncclConnector` | host 管理一条发送或接收连接：实现回调、私有资源、proxy 状态和 `connected` 标志 |
| `ncclConnInfo` | 给 GPU 使用的连接信息：buffer 地址、协议进度、FIFO 和相关参数 |

可以简化成：

```text
comm->channels[c].peers[peer]
  → send/recv[connIndex]：ncclConnector
       └ conn：ncclConnInfo
```

host 负责创建和维护资源，GPU 使用准备好的地址和状态。`connected=1` 之前，不能把逻辑邻居当成可用数据通道。

## 2. 通用建连流程

不同 transport 共享一个基本流程：

```text
canConnect：当前路径是否允许？
  ↓
选择 transport 并 setup：准备本端资源和描述
  ↓
bootstrap 交换描述
  ↓
connect：消费对端描述，完成映射或网络握手
  ↓
填写 ncclConnInfo，复制到 device，标记 connected
```

`ncclTransportP2pConnect` 先只登记需求：

```text
connectRecv[peer]：需要从谁接收
connectSend[peer]：需要向谁发送
```

`ncclTransportP2pSetup` 再统一完成 transport 选择、描述交换和实际连接。

**登记 bitmask 不等于连接已经建立。**

## 3. 三种常见 transport 怎样理解？

| Transport | 数据通路 | 主要准备内容 |
|---|---|---|
| P2P | GPU 经 NVLink/PCIe 访问另一块 GPU 的内存 | P2P 能力检查、CUDA IPC 或 cuMem 映射和访问权限 |
| SHM | 发送 GPU 写 host 共享缓冲，接收 GPU 读取 | 创建和导入共享内存对象，设置 GPU 可见地址 |
| NET | GPU/内部缓冲经 NIC 到对端 | 网卡与协议缓冲、监听/连接 handle、MR、proxy 状态 |

### P2P：导入的是访问关系，不是指针数值

不同进程的虚拟地址可以不同。拥有者先导出共享 handle，接收者导入后在自己的地址空间建立映射并设置访问权限。

所以，**只把一个 GPU 指针数值发给另一个进程，并不能保证对方可以访问它。**

### SHM：CPU 准备资源，GPU 搬数据

SHM 的 payload 位于主机内存，但常规数据读写仍可由 GPU 完成。proxy service 负责共享对象的创建、导入和释放，不能据此认为 payload 一定由 CPU 逐片搬运。

### NET：网络连接、GPU 缓冲和 MR 要接起来

NET 的典型过程是：

```text
接收侧 listen → 交换 handle → 发送侧 connect / 接收侧 accept
  → 准备协议缓冲和网络注册
  → 把地址与状态写入 conn
```

网络 MR 让 NIC 获得访问某段内存的资格；GPU 是否能访问同一区域则是另一个问题，需要分别建立。

## 4. GDR、MR、IPC 不要混为一谈

| 机制 | 解决谁的访问问题 |
|---|---|
| CUDA IPC / cuMem 映射 | 另一个进程或 GPU 如何访问某段 GPU 内存 |
| MR 注册 | NIC 如何获得访问一段内存的网络权限和句柄 |
| GDR | NIC 是否可以直接读写 GPU 内存，避免 host buffer 中转 |

```text
GDR + 内部 GPU 缓冲：
    GPU 协议槽 → NIC → 网络 → NIC → 对端 GPU 协议槽

使用 host 缓冲：
    GPU → host buffer → NIC → 网络 → NIC → host buffer → GPU
```

使用 GDR 时仍可能经过 NCCL 的内部协议缓冲，不能把它理解为所有数据都绕过 NCCL 直接访问用户数组。

## 5. 一条跨机边怎样建立？

假设 r1 的 `ring.next=5`，r5 的 `ring.prev=1`，选择 NET：

```text
r1：登记向 r5 发送的需求
r5：登记从 r1 接收的需求
  ↓
r5 的接收 service 创建监听端点并生成 handle
  ↓
bootstrap 将 handle 交给 r1
  ↓
双方推进 connect/accept，准备缓冲和 MR
  ↓
双方填写 ncclConnInfo，后续 GPU 与 proxy 复用连接
```

这次交接中有三种不同信息：

- **Ring 邻居**：决定和谁通信。
- **建连描述**：让两端找到并配置同一条连接。
- **设备连接信息**：让 GPU/proxy 后续知道 buffer 和进度在哪里。

## 6. 最后分清三件事

```text
逻辑邻居：r1 的后继是 r5
连接资源：NET handle、MR、映射和缓冲已经准备好
执行推进：GPU 与 proxy 按协议发布、等待和回收数据片
```

如果启用了 `runtimeConn`，这些资源可以在第一次选择该算法时再建立。多轮 collective 会继续复用，而不是每轮重新建连。

下一步读 [08｜Proxy 与网络](08-proxy-and-network.md)，看连接建立后，GPU、proxy 和 NIC 怎样共同推进一片数据。

需要 `ncclConnect` 的具体布局、P2P/SHM setup/connect、NET listen/accept 和 GDR 映射细节时，再读 [Transport 进阶附录](appendix/transports.md)。
