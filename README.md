# MHA精度与加速比自测

从Q4迁移的独立MHA验证仓库，仅保留固定上游源码、MHA用例、精度检查、计时和汇总，不包含mHC及历史性能结果。本仓库第一次NPU测量尚待执行，不把旧仓库结果当作本轮结果。

## 测试范围

- BF16输入、内核原生BF16输出，B=H=1、D=128，非因果、无dropout。不是默认FP32输出示例的验收。
- 整块：S=4096、3584；对比旧TL、PR静态TL、PR动态TL和AC。
- 非整块：S=1、63、64、65、127、129、193、255、3583、3585、4095、4097；仅比较PR动态TL和AC，不做Host补齐。
- 一个动态TL内核在每个进程中复用该组全部长度；编译不计时。
- TL旧版80364b3d9758854a9692cc0f14089025f54d3f3f；PR3896固定86dccfce8266818a7f6278de84b5c28b9851fa46；AC固定ba5162411c919e15fbabb28298e8b420c11ad905。
- 原始TL源码按原字节保存并校验SHA256；不重新安装TileLang编译器。

## 精度及计时

CPU端FP32计算参考Attention后转BF16。输出shape、dtype须匹配，无NaN或Inf，同时相对L2误差<=0.01、最大绝对误差<=0.02。检查每条实现与参考以及实现间一致性；所有检查通过才计时。该标准不同于上游test_mha.py的最大范数相对误差rel<0.05。

每规格3个独立进程，每进程预热100次、10轮、每轮100次。保存Event和wall耗时，轮次及进程均取中位数；轮换实现测量顺序。AC/TL>1表示TL更快。

AC仅测per_call：每次调用重新生成metadata，生成开销计入Event/wall耗时。不执行或汇总cached模式，不能把per_call差距当作纯Attention内核差距。每个长度测3个独立进程。

## NPU服务器重跑

沿用现有zwh环境，先加载本机实际安装的CANN set_env.sh。以下仅复用旧Q4已经安装的AC运行依赖，不读取其测试代码或历史结果。

```bash
conda activate zwh
cd /home/z30086261
git clone https://github.com/zhangwenhao91/Q4_test.git
cd Q4_test
bash setup_runtime.sh /home/z30086261/Q4
export ASCEND_RT_VISIBLE_DEVICES=4  # 修改为实际空闲卡
bash run.sh smoke
```

确认smoke精度成功后：

```bash
bash run.sh all
```

smoke覆盖全部14个长度，all重新执行两组完整检查和性能测量。结果位于唯一的results/mha-pr3896时间戳目录，脚本会输出目录。将两份summary.txt内容贴入本仓库error.txt并推送；出错贴runner.log完整异常。尽量同时提交metadata.json、correctness.json、timings.csv、summary.json、revision.txt、pip-freeze.txt、npu-smi.txt到captures目录，便于核实设备和原始数据。

setup_runtime.sh通过符号链接复用旧Q4的third_party/ops-transformer-q4fa和q4fa-opp，保留源码提交及已编译库SHA256校验。运行期间请保留这两个目录；不会复制整个旧仓库，也不会重装软件。若原生依赖已丢失，先恢复对应固定版本的q4fa安装，不用其他AC实现代替。

## 文件位置

| 文件 | 用途 |
|---|---|
| run.sh | 一键执行整块与非整块smoke/all |
| setup_runtime.sh | 关联已有AC源码及安装依赖 |
| operators/spec_v1/bench.py | 独立MHA精度与Event/wall计时框架 |
| operators/spec_v1/mha_bf16.py | 旧TL和AC调用封装 |
| operators/spec_v1/upstream_mha | 固定旧TL源码及许可证 |
| operators/mha/pr3896/bench_pr.py | PR静态、动态及非整块测试 |
| operators/mha/pr3896/upstream | 固定PR源码及许可证 |
| operators/mha/pr3896/summarize.py | 无NPU汇总计时与逐进程加速比 |

测试环境由新运行实际记录，不能假定为32核950DT。当前用例仅覆盖Q和KV等长，不覆盖独立sq/T、完整MHA投影层或反向。
