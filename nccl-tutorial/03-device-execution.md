# 03｜Device execution：CTA 怎样执行一份 work

上一章已经把请求变成 CUDA launch 参数。本章只回答一个问题：**GPU 上的一个 CTA 怎样找到自己的 channel，读取 work，并执行 Ring AllReduce？**

第一次读不必展开所有协议字段。先记住这条链：

```text
blockIdx.x → channelId → 装载 devComm/channel/work
           → devFuncId 分派 → RunWorkColl → runRing → Primitives
```

进阶协议、slice/step 和 FIFO 细节见 [Device 执行进阶附录](appendix/deep-dive/device-execution.md)。

## 1. 继续使用最小例子

```text
r0：[1,1,1,1]
r1：[2,2,2,2]
结果：[3,3,3,3]

Ring / Simple / 1 个 channel
```

host 已设置：

```text
channelMask = 0b0001
grid.x = 1
```

因此当前只有一个 CTA，它负责 channel 0 的全部 4 个元素。

## 2. 第一步：由 blockIdx 找到 channel

`channelMask` 是活动 channel 的位图，`blockIdx.x` 是 launch 中的连续编号。

| channelMask | 活动 channel | block 与 channel 的对应 |
|---|---|---|
| `0b0001` | 0 | block 0 → channel 0 |
| `0b0110` | 1、2 | block 0 → channel 1，block 1 → channel 2 |
| `0b1111` | 0、1、2、3 | 四个 block 分别对应四个 channel |

[`ncclKernelMain`](../nccl/src/device/common.h#L419) 会数当前 channel 前面有多少个置位，并将它映射为 `blockIdx.x`。

所以，**blockIdx 不等于 channel ID；它只是置位 channel 的序号。**

## 3. 第二步：装载三份执行信息

确定 channel 后，CTA 把执行所需的信息装入共享内存：

| 信息 | 来源 | 作用 |
|---|---|---|
| `ncclKernelComm` | 初始化时的 `comm->devComm` | rank 数、channel 数组、buffer 和执行环境 |
| `ncclDevChannel` | 初始化时的 channel 规划 | 当前 channel 的 Ring/Tree 邻居和 peer 连接 |
| `ncclDevWorkColl` | 本次任务 | 用户 buffer 地址、数据分工、chunk 和规约参数 |

可以把它们理解成：

```text
devComm：整个通信组的情况
channel：这条并行通道的连接关系
work：这次要处理的数据
```

payload 仍留在用户 buffer 中，不会随这些描述一起复制到共享内存。

## 4. 第三步：根据 devFuncId 分派工作

随后，kernel 根据 batch 中的 `funcId` 选择执行入口：

```cpp
if ( SpecializedFnId matches ncclShmem.funcId )
  SpecializedRunWorkBatch().run();
else
  ncclDevFuncTable[ncclShmem.funcId]();
```

本例会进入：

```text
RunWorkColl<AllReduce, float, Sum, Ring, Simple>
```

`RunWorkBatch` 再遍历 batch 中的 work，调用相应的 `RunWorkColl::run`。

这里要区分：

- `kernelFn`：host 启动哪个 CUDA kernel 入口。
- `devFuncId`：device 内部具体执行哪种 collective/algorithm/protocol 工作。

同一个通用 kernel 可以承载不同工作，实际算法要看 work 的 `funcId`。

## 5. 第四步：`runRing` 把三者接起来

[`runRing`](../nccl/src/device/all_reduce.h#L14) 开头做三件事：

### ① 读取 Ring 位置

```cpp
ncclRing* ring = &ncclShmem.channel.ring;
int ringIx = ring->index;
int nranks = ncclShmem.comm.nRanks;
```

`ringIx` 是当前 rank 在环中的位置，邻居来自初始化结果。

### ② 恢复数据分工

```cpp
ncclCollCbdPart(...);
```

它把 host 写入的 channel 划分恢复成：

- `gridOffset`：从用户数组哪个元素开始；
- `channelCount`：当前 channel 处理多少元素；
- `chunkCount`：每个逻辑 chunk 处理多少元素。

本例可以理解为从 0 开始处理全部 4 个元素。

### ③ 创建 Primitives

`Primitives` 接收：

- `ring.prev/next`：从前驱接收、向后继发送；
- `work->sendbuff/recvbuff`：用户输入输出地址；
- 规约参数。

于是，`runRing` 不需要知道所有 transport 细节，只需按照 Ring 顺序调用收发和规约原语。

## 6. 两个数学阶段在同一个工作体中完成

Ring AllReduce 的逻辑是：

```text
ReduceScatter：先让每个 rank 得到一个完整规约块
AllGather：再把这些完整块传播给所有 rank
```

用两 rank 演示：

```text
R0：C0=[1,1]，C1=[1,1]
R1：C0=[2,2]，C1=[2,2]
```

1. R0 先把 C1 发给 R1，R1 先把 C0 发给 R0。
2. R0 将收到的 R1 C0 与本地 C0 相加，得到 `[3,3]`；R1 同样得到完整的 C1 `[3,3]`。
3. 双方再把完整块发给对方，最后都得到 `[3,3,3,3]`。

这些动作在同一个 `runRing` 中完成，不需要中间返回 host API。

## 7. Primitives 为什么还需要协议状态？

算法决定“这一跳该收发什么”，但实际读写还要满足两个条件：

```text
发送方：目标槽位是否可写？
接收方：数据是否已经可读？
```

Simple 用连接上的进度计数表达：

- `tail`：生产者已经发布到哪里，接收方据此判断数据是否可读。
- `head`：消费者已经消费到哪里，发送方据此判断槽位能否复用。

可以先记成：

```text
tail 管“新数据能不能读”
head 管“旧空间能不能覆盖”
```

`credit` 就是消费者归还给生产者的可写空间。下游不前进时，槽位会逐渐占满，上游便等待，这就是背压。

## 8. 本章主线复习

```text
blockIdx + channelMask
  → channelId
  → 装载 devComm/channel/work
  → devFuncId 分派到 RunWorkColl
  → ncclCollCbdPart 恢复数据分工
  → Primitives 使用 ring.prev/next 和用户 buffer
  → runRing 完成 ReduceScatter + AllGather
```

下一步读 [04｜集合算法](04-algorithms.md)，用完整的四 rank 例子理解 Ring 为什么不会漏加或重复加贡献。

需要研究 CTA 内线程角色、chunk/slice/step、LL/LL128 和 FIFO 可见性时，再读 [Device 执行进阶附录](appendix/deep-dive/device-execution.md)。
