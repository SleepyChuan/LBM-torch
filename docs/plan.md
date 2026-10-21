# 实施计划

按顺序做，每个阶段结束时代码都能跑、测试都能过。不要跳阶段——
LBM 的 bug 最恶心的地方是"跑起来了、看起来也对"，等到 Poiseuille 对不上
再回头找，根本不知道是哪一层错了。

> 状态：P0~P8 已全部落地。TRT（`collision="trt"`，默认）与 Zou/He 入口出口
> 是 P8 之后追加的：参数见 `architecture.md`，物理见 `physics.md`。

## P0 骨架与工厂

建 `pyproject.toml`（必需 `torch>=2.3`、`numpy`、`tqdm`；可选 `pytest`、`h5py`、
`pyvista`、`matplotlib`）。写 `LatticeSpec` 数据类、`d2q9.py` 的方向表、三个
solver 的类和工厂。D3 的两个类构造能成功，`initialize` / `step` 开头的
`_check_implemented()` 抛 `NotImplementedError`。

工厂对未知模型名抛 `ValueError`，和"未实现"区分开。

验收：`create_solver("D2Q9", (8,8), nu=0.1)` 返回实例；
`create_solver("D3Q19", (8,8,8), nu=0.1).step()` 抛 `NotImplementedError`；
`create_solver("D2Q8", (8,8), nu=0.1)` 抛 `ValueError`。
`test_stencils.py` 通过（c/opp/w 的自洽性）。

## P1 平衡态与宏观量

`equilibrium.py`、`macros.py`。都用广播写，不区分模型。

验收：`test_equilibrium.py` 用随机 `rho ∈ [1.0,1.1]`（代码里 `1 + 0.1*rand`）、
`|u| < 0.1` 的场，检查三个矩条件，fp32 下零阶矩误差 < 1e-5、一阶/二阶矩
与对称性 < 1e-6。另外检查 `u` 取反时 `feq` 满足 `feq_i(u) = feq_opp_i(-u)`。

## P2 BGK 碰撞（无力）

`CollisionOperator` 协议 + `BGK(tau)`。此时还没有 streaming，测试直接
对张量调用碰撞算子。

验收：`test_collision.py`——碰撞前后 `rho`、`rho*u` 不变（<1e-6）；
输入本身就是 `feq` 时输出不变（不动点）；直接验解析关系
`f_post - feq == (1 - 1/tau) * (f - feq)`（不要用 `|f_post - feq|` 比大小：
`1 - 1/tau` 在 `tau = 1` 处过零，见 `pitfalls.md`）。

## P3 pull streaming + 周期

`BaseSolver` 的双缓冲、`_advance`、`D2Q9Solver._stream`。
**先只写无固体的纯周期版本**，`solid=None`。这一步不加反射。

验收：`test_streaming.py`——域内均匀 `rho=1, u=(0.02, 0)` 的平衡态场，
跑 1000 步后逐点与初值相同（<1e-6）。

这个测试是整个项目里最有价值的一个。它同时会抓到：`opp` 表错、
方向表把 `+x` 和 `+y` 编号搞反、切片方向写反、双缓冲下标错、feq 公式错。
只要它不过，就不要往下走。

## P4 固体与半程反弹

预计算 9 个 `wall_mask` 和 `_has_wall` 标志，`_stream` 里加反射覆盖。
`macros`/`diagnose` 排除固体节点。碰撞结果只写流体节点。

验收：`test_streaming.py` 续——静止流体（`rho=1, u=0`）加一个固体块，
跑 1000 步后仍然静止（<1e-6），总质量守恒（<1e-7）。
这个测试之所以成立，是因为 `u=0` 时 `feq_i = feq_opp_i`，
反射分支和正常分支给出同样的值。

另外测：`solid` 全为 `True` 的退化域不崩、不产生 NaN。

## P5 Guo forcing

`BGK` 加源项，`macros` 加 `F/2` 修正。`force=None` 时整条路径跳过。

验收：`test_forcing.py`——全域周期、无固体、`u0=0`、`F=(1e-5,0)`，
跑 100 步后 `ux ≈ (100 + 0.5) * 1e-5 = 1.005e-3`（注意那个 `+F/2`），
用绝对容差 `< 1e-6`。单步（n=1）时 `ux = 1.5F`，噪声极小，可以卡到 `1e-7`。

这个测试能单独抓出 `F/2` 修正和 `(1 - 1/(2*tau))` 因子的问题：
少写前者，稳态速度偏 1/2；少写后者，偏差随 `tau` 变化。

## P6 诊断与 IO

`diagnostics.py`、`io/npz.py`、`dump_setup`。h5/vtk 只写签名。

验收：`test_io.py`——json 字段齐全且可解析；npz 文件名带步数；
读回的 `pdf` 形状 `(Q,Nx,Ny)`、dtype 等于 `output_dtype`；
推进若干步后文件内容确实变了。

## P7 Poiseuille 验证

`examples/poiseuille.py`，参数和容差见 `validation.md`。

验收：跑通 fast 档，误差在表内。再跑一次收敛性检查，记录观测阶。

## P8 compile 与性能基准

`examples/bench.py` 输出 MLUPS（`Nx*Ny*steps / 1e6 / seconds`）。
在 CUDA 上对比 `compile=True/False`，确认结果一致、速度有提升。

验收：CUDA 上 `compile=True` 与 eager 跑同样 1000 步，场差异 < 1e-3；
MPS 上如果编译不可用，明确报错并在 `pitfalls.md` 里记下结论。

## 编码约定

类型注解写全，`torch.Tensor` 用 `from __future__ import annotations` 省掉引号。
所有常量张量在 `__init__` 里建好放 device 上，`_advance` 里不新建张量
（除了必然存在的中间量）。

日志用 `logging`，不要在库里 `print`。

测试用 `pytest`，Poiseuille 和收敛性测试标 `@pytest.mark.slow`，
默认跑 `pytest -m "not slow"`。

## 不要做的事

不要提前做多卡。不要为了"以后可能"把 `_stream` 抽成通用函数——那正是
用户明确说不要的过度抽象。（TRT 与 Zou/He 已在 P8 之后实现，见
`physics.md`；D3 仍然只留元数据。）

不要引入 `jax`、`warp`、`numba` 做混合加速，v1 就是纯 PyTorch。

不要在热循环里放 `logger.info`，日志在 `step()` 外面打。
