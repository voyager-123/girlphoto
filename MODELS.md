# 模型 1 / 模型 2：先搭程序，再接真实数据

本目录提供 PyTorch 小模型、训练入口、可求导调色引擎、数据检查和预测导出。
**没有提供经过真实人像数据训练的权重。合成演示只检验程序流程，不代表学会了审美。**
模型 2 已接入本地调色测试网页，也可以通过命令行训练和预测。

## 环境完全分开

- 标注工具沿用 `start.bat`。
- 模型只用项目内 `.venv-models`，关闭主环境包共享，不向系统 Python 安装依赖。
- `setup_models.bat`、`models.bat`、`demo_models.bat`、`start_try.bat` 都明确调用该虚拟环境。
- 不需要手动激活环境。也不要用裸的 `pip install` 安装模型依赖。
- 如果已经激活 `.venv-models`，也可以直接运行安装脚本：它会复用已有环境，不会覆盖正在运行的 Python。
- 脚本安装前会检查环境位置和隔离设置；可用 `setup_models.bat --check` 只检查环境、不下载依赖。
- 环境、权重和 `ml_runs/` 中的训练图片、日志均被 Git 忽略。

第一次，在项目终端执行：

```powershell
.\setup_models.bat
```

默认安装 CPU 版 PyTorch，不需要 CUDA 或显卡。若下载需要你当前的本地代理，可以执行：

```powershell
.\setup_models.bat http://127.0.0.1:10792
```

代理仅对安装脚本这一进程有效，不修改系统或 Git 的代理配置。端口须以你实际使用的代理软件为准。
看到 `Ready` 后再运行模型。下载失败不表示 Python 版本不兼容，也不应把包改装进主环境。

## 一次跑完整条演示流程

```powershell
.\models.bat demo --out ml_runs\demo01 --steps 30
```

也可以双击 `demo_models.bat`，默认输出到 `ml_runs/demo`。
重复实验请指定新的输出目录，避免覆盖此前的模型。

程序会：

1. 生成六组明确标记为合成的数据，四组训练、一组验证、一组测试。
2. 用人工设定的演示标签训练模型 1。
3. 用已知目标参数、目标图以及冻结的模型 1 反馈训练模型 2。
4. 保存 `model1.pt`、`model2.pt` 和各自的 `.metrics.json`。
5. 对保留的验证场景输出 `preview.png` 与 `preview.parameters.json`。

验证集不参与梯度更新。这里的数据很少，训练步数也短，结果只能证明流程是否能执行。
模型文件会保存 `demo_only=true`；程序拒绝让演示评分器指导标记为真实数据的模型 2 训练。

## 用网页测试自己的照片

双击 `start_try.bat`，浏览器打开 <http://127.0.0.1:8765/try>。
如已有标注服务占用该端口，先关闭旧服务，或执行 `start_try.bat --port 8766`。

1. 选择 JPEG、PNG 或 WebP 原图（静态图片，最多 25 MB、1600 万像素）。
2. 选择模型 2 权重、目标风格及 0–100% 强度，点击“生成调色结果”。
3. 对照原图检查肤色、明暗和颜色，查看六项参数，下载预览图或参数 JSON。

网页自动发现 `ml_runs/` 下本项目格式的模型 2 权重，只展示该权重训练过的风格。
添加或重新训练权重后，点击“刷新模型”。演示权重有醒目标记；其他权重也不表示效果已验证。
可以切换监督版与反馈版测试同一张照片，目前不会自动保存各次结果，请自行下载留存。

上传照片仅在本机内存中处理，不保存到标注库、不用于训练；下载文件由浏览器保存。
预览经过方向及色彩处理，最长边 1600 像素。零强度保持处理后的原图预览，不等于复制原文件字节。
模型从上传原图读取特征，六项参数属于本项目调色引擎；全尺寸输出仍使用下文的预测命令。
参数 JSON 记录模型与原图的 SHA-256、风格、强度及尺寸，便于追踪测试版本。
页面不使用模型自己的质量分数作为“效果好”的证明，实际效果仍需人工判断。

## 两个模型分别学什么

### 模型 1：评价器

小型卷积编码器共享图像特征，但输出彼此独立：

| 输出 | 训练数据 | 含义 |
| --- | --- | --- |
| 多标签风格 | 每类 yes / no，未知屏蔽 | 是否具有某种调色风格，可同时符合多个风格 |
| 质量三分类 | 1 / 2 / 3，缺失屏蔽 | 颜色、明暗和处理协调性；不是人物长相 |
| 各风格强度 | 每类 0–1，可不提供 | 0 表示明确不存在该风格；未知不填 0 |
| 条件偏好分数 | 原图、同风格的两个版本、左右优劣或平局 | 给定原图和目标风格，哪个处理更合适 |

偏好头同时看原图特征、修图特征和目标风格；左右得分差用于偏好训练。
平局按目标概率 0.5 处理，跳过不参与训练。它不会假设所有场景都应该选暖黄或冷绿。
不同输出按各自有效标签计算损失，未知风格和没有填写的强度不会被当成 0。
没有训练过的强度头在评分命令中返回 null，不显示成可信强度；风格概率也未做置信度校准。

### 模型 2：参数预测器

输入：原图 A、目标风格 S、强度 t。输出：六个归一化参数 P。
调色引擎执行 `A′ = render(A, P)`，模型 1 可对 A′ 提供反馈。

参数是曝光、对比度、冷暖、绿/洋红色调、饱和度、暗部调整。
范围被限制在 `[-t, t]`，t=0 时严格输出中性参数。**强度控制可用，不保证人的感知随 t 线性变化或每张图都单调变化。**

两种训练方式都已提供代码：

- **监督训练**：给目标参数或同原图的目标修图版本，不需要模型 1。
- **评分反馈训练**：加载已训练模型 1，并冻结其权重。梯度仍能通过评价器和调色引擎传到模型 2。可以同时使用监督数据。

评分反馈包含条件偏好、已训练的风格/质量/强度头，并加入小幅度的参数和画面变化约束。
各目标可能冲突，默认权重只是起点；低损失不能证明照片好看。
为避免未训练的偏好头误导模型 2，启用反馈时，目标风格必须有非平局偏好训练记录。
这只是最低可用性检查，不等于评分器质量合格；仍需独立人工评估。

## 之后喂数据：统一清单格式

模板在 `examples/training_manifest.example.json`。这是格式示例，里面的文件路径和评分须替换。
`demo` 自动生成的 `manifest.json` 是可运行的完整例子。

- `styles`：风格 ID 列表；两个模型必须使用完全相同的顺序。
- `samples`：照片 ID、路径、原图组、拍摄批次、train/validation/test、用途及可选标签。
- `pairs`：source 原图 ID、left/right 版本 ID、style 与 outcome（left/right/tie/skip）。
- `tasks`：模型 2 的 source、style、strength，加可选的 target 图片 ID 或 parameters。

`file` 可以是相对清单文件的路径，也可以是绝对路径。`role=original` 须由你明确指定；不能仅因文件未经标注就当成原图。
相关原图与修图必须共享原图组、拍摄批次和数据划分，程序会拒绝跨集合关联。
同一个人跨批次、近重复照片等仍需人工检查。

模型清单的 strengths 使用 0–1；如果以后标注工具采用 0/1/2/3 等级，需要明确转换规则，不能直接混用。
质量一直使用 1/2/3；示例里没填的分数和强度不是负例。

`tasks.parameters` 是按以下固定顺序填写的六个 **本引擎归一化参数**：

```text
[exposure, contrast, warmth, tint, saturation, shadows]
```

值须在 `[-strength, strength]` 内，全部为 0 表示不调整。它们不是 Lightroom 或其他软件的滑块数值。
没有本引擎参数时可以只填 target；全局调色无法复现美颜、磨皮、局部蒙版或改变人物形状等操作。
如果只有 source/style/strength，必须先有经过验证的模型 1，再启用反馈训练。

## 使用现有标注工具的数据

导出标注 ZIP 后，可转换某一位标注者的数据：

```powershell
.\models.bat convert-export --export exports\girlphoto-labels.zip --data-dir data --reviewer "我" --out ml_runs\real\manifest.json
```

默认不猜原图角色，不填造强度和参数。缺少来源原图的偏好对暂不转换，会报告数量。
如果你明确按 `001__original.jpg` 命名原图，可以追加：

```powershell
--original-marker __original
```

这是你指定的命名规则，转换后仍需人工核对。一个组匹配出两张原图会报错。
转换器读取 v1 标注导出，排除合成演示、跳过的标注和不可用比较，保持既有分组与划分；不移动原始图片。
生成的 tasks 为空，需要你随后填写模型 2 的任务。首次训练前执行：

```powershell
.\models.bat check-data --data ml_runs\real\manifest.json
```

## 训练和预测命令

先训练模型 1：

```powershell
.\models.bat train1 --data ml_runs\real\manifest.json --out ml_runs\real\model1.pt --steps 500
```

模型 2 先使用目标参数或目标图做监督训练：

```powershell
.\models.bat train2 --data ml_runs\real\manifest.json --out ml_runs\real\model2.pt --steps 500
```

在模型 1 的独立偏好验证通过后，可以加载它训练一个单独版本：

```powershell
.\models.bat train2 --data ml_runs\real\manifest.json --scorer ml_runs\real\model1.pt --out ml_runs\real\model2_feedback.pt --steps 500 --feedback-weight 0.2
```

这些命令从头训练，没有断点续训功能。`--steps` 是优化更新次数，不是遍历全部数据的轮数。
相同输出路径会覆盖模型，请给实验分配不同文件名。默认分辨率 96，两个模型反馈训练时分辨率必须一致。
默认 CPU；只有该独立环境另行装好 CUDA 版 PyTorch 且显卡可用时才选 `--device cuda`。

评分与独立测试：

```powershell
.\models.bat score --model ml_runs\real\model1.pt --image "D:\photos\sample.jpg"
.\models.bat evaluate1 --model ml_runs\real\model1.pt --data ml_runs\real\manifest.json --split test
```

生成预览与参数：

```powershell
.\models.bat predict --model ml_runs\real\model2.pt --image "D:\photos\original.jpg" --style retro --strength 0.7 --out ml_runs\real\result.png
```

输出 PNG 保持读入并按 EXIF 转正后的尺寸，参数保存在相邻的 `result.parameters.json` 中。
处理使用 sRGB，保留源文件，但输出不复制原文件 EXIF。参数记录包含引擎版本、实际换算值、风格、强度和演示标记。
当前只有全局调整，无脸部/肤色分割、局部调整或真实人像预训练。

## 验证

```powershell
.\.venv-models\Scripts\python.exe -m unittest discover -s tests_models -v
```

测试覆盖中性参数、零强度、参数范围、可求导反馈、缺失标签屏蔽、跨集合数据拦截、权重读写及双模型短训练。
测试通过后，再用合成演示验证命令行完整流程；真正效果必须靠保留场景和独立人工偏好评价。

2026-10-05 新增 5 项网页接口测试：模型发现、预测及零强度、预览尺寸、无效输入与权重变化、HTTP 令牌校验与标注库隔离。加上原有测试共 22 项通过。

2026-10-04 已在 `.venv-models` 中完成验证：模型的 7 项测试和原标注工具的 10 项测试均通过。
环境为 Python 3.12、PyTorch 2.14.1+cpu、NumPy 2.5.3、Pillow 12.3.0；未使用主环境安装依赖。

完整演示输出在 `ml_runs/verified-20261004/`：模型 1、2 各训练 30 步，保存并重新加载权重，
生成了 `preview.png` 与 `preview.parameters.json`。目录被 Git 忽略，可在本机查看。
模型 1 的演示训练损失从约 1.24 降至 0.38，模型 2 从约 0.30 降至 0.22。
这些数值只用于记录合成流程的实际运行，不是对真实照片的审美效果评估。
