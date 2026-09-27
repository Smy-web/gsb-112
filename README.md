# 寺庙修缮募捐功德纪要系统 · AI 服务层（转写 / 说话人分离 / 归属 / 文案生成）

## 这个仓库是什么

寺院要修缮殿堂。客堂开完现场会，把录音传上来，后台依次做四件事：语音转写、说话人分离、把转写段落归属到说话人、再让大模型认一认每个说话人是什么身份；认完之后再生成会议纪要、募捐文案、功德回向清单三段正文，最后交给渲染层出功德纪要 Markdown，存档、贴内网、月底群发给所有随喜过的信众。

这里裁剪出来的只有这条链路的**服务层**：`backend/services/ai_service.py` 里的 `AIService`，加上它依赖的 `backend/config.py`。

**不在范围内、也不要去找**：Markdown 渲染与邮件发送（`backend/services/document_service.py`）、REST 接口与 ORM（`backend/api/`、`backend/models.py`、`backend/database.py`、`backend/schemas.py`）、上游入口 `backend/main.py`、`frontend/`。上游怎么调用这一层，见「上游怎么调用」一节，够你判断兼容性了。

**本仓库的 README 是这个文件对外承诺的行为规格，以 README 为准，不要以现有代码为准。** 现有代码里有一批已经造成线上后果的偏差，下面 T1–T8 逐条写了契约；你的任务是补一套能钉死这些契约的测试，并把测试暴露出来的偏差修掉。

## 目录

- `backend/config.py` — 配置项，pydantic-settings，import 时从环境变量和 `.env` 读。本次不要动它既有的配置键。
- `backend/services/ai_service.py` — 本次要改的文件。`AIService` + 模块级 `SPEAKER_NAMES` + 模块级单例 `ai_service`。
- `pytest.ini` — 已经开了 `asyncio_mode = auto`，不要改。

`backend` 与 `backend.services` 都带空的 `__init__.py`，是常规包。验证命令必须从仓库根目录、用 `-m` 的形式跑，这样仓库根才会进 `sys.path`，`from backend.services.ai_service import AIService` 才解析得到。

## 运行环境

寺院客堂的内网机器，离线，没有配大模型密钥。`~/venvs/gsb-temple` 里本题用得到的只有 `openai 1.51.0`、`pydantic-settings 2.5.2`、`python-dotenv 1.0.1`、`pytest 8.3.3`、`pytest-asyncio 0.24.0`（完整 pin 见 `requirements-task.txt`）。

没有装 `pyannote.audio`、`whisper`、`torch`、`uvicorn`、`email-validator`，也不允许装。**`diarize_speakers` 里那句 `from pyannote.audio import Pipeline` 在本环境必然 `ImportError`**——这不是意外，这正是 T3 要钉的那条路径。

不要设置 `OPENAI_API_KEY`，不要联网，不要发出任何真实请求。所有大模型调用一律用标准库写的替身客户端顶掉（见「测试要求」）。

`openai` 必须是 1.51.0 配 `httpx` 0.27.2。httpx 0.28+ 去掉了 `proxies` 形参，构造 `AsyncOpenAI(...)` 会直接 `TypeError`，而模块末尾就有一句 `ai_service = AIService()`，于是整个模块连 import 都过不去。共享 venv 里已经是这个组合，不要动。

## 上游怎么调用（硬约定）

上游 `backend/api/meetings.py` 里那一整段后台处理（本次不携带，仅供判断兼容性）大致是：

```python
try:
    transcript_result = await ai_service.transcribe_audio(meeting.audio_file_path)
    meeting.transcript = transcript_result["transcript"]

    speaker_segments = await ai_service.diarize_speakers(meeting.audio_file_path)
    meeting.speaker_diarization = {"segments": speaker_segments}

    aligned_segments = ai_service.align_transcript_with_speakers(
        transcript_result["segments"], speaker_segments)

    speaker_roles = await ai_service.identify_speaker_roles(aligned_segments)

    for seg in aligned_segments:
        speaker = seg["speaker"]
        db.add(models.DialogueTurn(
            meeting_id=meeting_id, speaker=speaker,
            speaker_role=speaker_roles.get(speaker, "未知"),
            content=seg["text"],
            timestamp=f"{seg['start']:.1f}s - {seg['end']:.1f}s"))

    meeting.summary = await ai_service.generate_meeting_summary(meeting.transcript, hall_info)
    meeting.fundraising_copy = await ai_service.generate_fundraising_copy(meeting.summary, hall_info)
    meeting.merit_list = await ai_service.generate_merit_list(meeting.fundraising_copy)
    ...  # 交给 document_service 渲染、落盘
    meeting.status = "completed"
    db.commit()
except Exception as e:
    meeting.status = "failed"
    db.commit()
    print(f"处理会议 {meeting_id} 时出错: {str(e)}")
```

从这段能读出四条硬约定，改的时候别踩：

1. `transcript_result["transcript"]` 和 `transcript_result["segments"]` 是裸下标取的；`segments` 里每一项的 `start` / `end` / `text` 也是裸下标取的。
2. `speaker_segments` 被原样塞进 `meeting.speaker_diarization` 落库，前端时间轴直接画它。
3. `speaker_roles.get(speaker, "未知")` —— 字典里没这个键时上游自己兜底成 `未知`；`seg["start"]` 要能被 `f"{x:.1f}"` 格式化。
4. 整段包在 `try/except Exception` 里，异常只 `print` 一行，然后 `status="failed"`。**抛异常是上游认得的失败信号，返回一个看起来像成功的值是上游认不出来的。**

---

## AI 服务层契约（backend/services/ai_service.py）

**T1 失败不许伪装成正文，凭据不许外泄。** 现状是三个 `generate_*` 都写成 `except Exception as e: return f"生成…时出错: {str(e)}"`。openai SDK 抛出来的 `e`，`str()` 之后长这样：

```
Error code: 401 - {'error': {'message': 'Incorrect API key provided: sk-abc1****xyz', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}
```

**里面带着密钥片段。** 这句字符串被上游当成正文写进 `meeting.summary` / `meeting.fundraising_copy` / `meeting.merit_list`，再渲染进功德纪要，存档、贴内网、月底群发给所有随喜过的信众。运维在 `.env` 里手滑写错一个字符，结果是几百封邮件里印着报错串和半截 key。

契约：

- 三个方法的返回值只可能是能拿给信众看的正文。失败必须以调用方能识别的方式暴露，不许再混进返回值。
- 任何情况下不得把 `settings.OPENAI_API_KEY`、`settings.OPENAI_BASE_URL`、请求头、或异常文本里的凭据片段写进返回值、日志、或任何落盘产物。脱敏要在异常文本离开这个方法之前做完。

> 用哪种方式暴露失败：抛一个自定义异常（类名你定，须继承 `Exception`、带可读原因），还是返回一个带 `ok` / `error` 键的结构体——两种都符合契约，选定一种并写明。选结构体就等于改了返回类型，上游那三处 `meeting.summary = await ...` 会存进去一个 dict，你要写清调用方该怎么改；选异常则上游那段 `try/except Exception` 会自然走到 `status="failed"`。三个方法必须用同一种。
> 选定方案：（待填）

**T2 `transcribe_audio` 的失败必须让上游停得下来。** 现状失败返回 `{"transcript": "", "segments": [], "error": str(e)}`，成功返回 `{"transcript": ..., "segments": [...]}` ——**成功那一边没有 `error` 键**。上游从不检查 `error`，拿着空串一路往下走：摘要、文案、回向清单全是对着空转写生成的，最后 `status="completed"`。运维看到的是「这场会处理完了」，点开纪要是一片空。音频被清理掉、路径写错、配额用完，表现全都一样。

契约：

- 成功返回的形状固定：`{"transcript": str, "segments": [{"start": float, "end": float, "text": str}, ...]}`。键名不许改，每个 segment 的三个键名也不许改。
- 失败必须让上游能停下，不许再返回一个与成功同形、无从区分的空结果。
- 失败原因同样要过 T1 的脱敏。

> 失败怎么表达：直接抛自定义异常（与 T1 同一种），还是让 `error` 键**恒在**——成功时值为 `None`、失败时为非空的可读原因字符串，由上游判断。两种都符合契约，选定一种并写明。选后者就等于要求上游加一句 `if transcript_result["error"]: ...`，写清楚。
> 选定方案：（待填）

**T3 `diarize_speakers` 不许把哨兵混进分段列表。** 现状 pyannote 不可用时返回 `[{"error": "No module named 'pyannote'"}]` —— 一个只含 `error`、没有 `start`/`end`/`speaker` 的字典，混在「说话人分段」列表里。上游 `meeting.speaker_diarization = {"segments": speaker_segments}` 原样落库，前端时间轴于是多出一段画不出来的「分段」，标注同事拿到 JSON 还得先猜哪个键在。本环境没装 pyannote，这条路径必然触发。

契约：

- 返回值恒为「说话人分段」列表，每条形如 `{"start": float, "end": float, "speaker": str}`。列表里不得混入 `error` 字典。
- 分离能力不可用（依赖没装、模型下不下来、鉴权失败）时返回**空列表**，另以某种方式让运维知道原因。
- 空列表必须能被下游安全消化：`align_transcript_with_speakers` 拿到空列表时，每条转写段仍要出一条结果，`speaker` 为 `"UNKNOWN"`（这是现状，属 preserve，要有测试钉住）。

> 「另以某种方式告知」用哪种：标准库 `logging` 记一条 warning、抛自定义异常、还是在实例上留一个可用性标记（如 `self.diarization_available`）——三种都符合契约，选定一种并写明。注意上游现在没有一处检查这个：选抛异常，就等于让整条处理链在每一台没装 pyannote 的机器上直接 `status="failed"`，转写和文案全都不做了。这个后果要不要接受，写清楚。
> 选定方案：（待填）
**T4 `identify_speaker_roles` 的返回值必须落在白名单里。** 现状是 `role = response.choices[0].message.content.strip()` 之后原样 `speaker_roles[speaker] = role`。提示词里写了「请只返回角色名称」，但模型实际会回一整句——`这个人应该是负责木作的老工匠师傅，语气比较像住持` 也照存。这句话经上游 `speaker_role=speaker_roles.get(speaker, "未知")` 写进 `dialogue_turns.speaker_role`，再被渲染层印成 `> **SPEAKER_00（这个人应该是负责木作的老工匠师傅，语气比较像住持）** [12.0s - 18.0s]：……`，进了群发邮件，也进了前端那个按角色筛选的下拉框——下拉框里于是多出十几个谁也认不出的「角色」。异常时静默降级成 `"未知"`，日志里一句都没有，运维无法区分「模型判不出来」和「请求根本没成功」。

契约：

- 返回值是 `Dict[str, str]`：键是 `aligned_segments` 里出现过的 `speaker`，值必须落在白名单 `{住持, 监院, 工匠, 设计师, 居士, 其他}` 之内。
- 模型回的整句、白名单外的词、空串、只有标点、带换行和多余空格，一律归一到白名单内的某一个值。
- 请求失败时不得静默：必须用标准库 `logging` 记一条 warning，写清是哪个 speaker 失败的、失败原因是什么。原因里同样不得含凭据（见 T1）。

> 两处要你定：① 白名单外的值归到 `其他` 还是保留上游那个 `未知` 字样（`未知` 不在白名单里，选它就等于把它加进白名单，写清楚）；② 从模型回的整句里抠白名单词，是按白名单固定优先级取第一个命中的、按在句子里出现的先后取、还是干脆只在整句**恰好等于**白名单词时才认——抠不出来时归到哪里，一并写明。
> 选定方案：（待填）

**T5 `UNKNOWN` 不发请求。** 现状 `align_transcript_with_speakers` 给分不到说话人的段落打 `"UNKNOWN"`，`identify_speaker_roles` 把它当成一个真说话人，照发一次请求，问「说话人UNKNOWN的身份角色」。离线场景下分离必然失败（T3），于是**每一场会都白发一次请求**：钱和时延白花，回来的角色还要被印进纪要。

契约：

- `"UNKNOWN"` 不发大模型请求。它在返回的字典里要么不出现（上游 `.get(speaker, "未知")` 会兜住），要么固定映射到一个不发请求就定下来的值——二选一，写明。
- 测试要能统计实际发出的请求次数：替身客户端必须记录每一次 `chat.completions.create` 调用，包括调用次数、`model`、`temperature` 和实际发出的 `messages`。

**T6 `align_transcript_with_speakers` 的输入容错与输出形状。** 现状 `trans_seg["start"]` / `["end"]` / `["text"]` 是裸下标取。上游传的是 `transcribe_audio` 的产物，但转写段也可能来自别的通道（人工补录、旧版本 SDK 少给一个键、JSON 往返过一次）。少一个 `text` 就是 `KeyError: 'text'`；`start` / `end` 是字符串 `"12.5"` 就是 `TypeError: unsupported operand type(s) for -: 'str' and 'float'`。这两种异常都被上游那段 `except Exception` 吞掉，整场会 `status="failed"`，日志里只有一句 `处理会议 3 时出错: 'text'` ——运维拿着一个引号里的单词，什么也查不出来。

契约：

- 输出每条形如 `{"start": float, "end": float, "speaker": str, "text": str}`，四个键名固定；`start` / `end` 必须是能被 `f"{x:.1f}"` 格式化的数（上游拿它拼 `timestamp`）。
- 合法的转写段按输入顺序输出，不许重排、不许合并、不许丢。
- `speaker_segments` 里混进来的 `error` 字典必须跳过，不参与重叠计算（现状即如此，属 preserve，要有测试钉住）。
- 输入不合法（缺 `text`、缺 `start`/`end`、时间不是数）时，不许让裸 `KeyError` / `TypeError` 传到调用方。

> 不合法的那一段怎么办：跳过它并 `logging` 记一条 warning（其余段照常出结果，输出条数会少于输入，上游就少建一条 `DialogueTurn`），还是抛一个带可读原因的自定义异常（整场会失败，但日志里能看出是第几段、缺哪个键）——两种都符合契约，选定一种并写明。选跳过，就要接受「一场会少几句发言也没人发现」；选抛，就要接受一句脏数据毁掉整场会。
> 选定方案：（待填）

**本次明确不在范围**：重叠为 0 时归 `"UNKNOWN"`、平局时按输入顺序取胜（把 `speaker_segments` 反转过来结果就变）、`overlap > max_overlap` 用的是严格大于——这套切分与平局算法**保持现状**，别改，也别拿它当验收点。另有排期。

**T7 `SPEAKER_NAMES` 要覆盖任意 `SPEAKER_<数字>`。** 现状模块级 `SPEAKER_NAMES` 只写了 `SPEAKER_00` 到 `SPEAKER_04`（甲乙丙丁戊），而 `identify_speaker_roles` 用 `SPEAKER_NAMES.get(speaker, speaker)`。pyannote 分出第六个人是常事，于是裸 id 被塞进提示词：「判断说话人SPEAKER_05的身份角色」。模型对着一个英文编号猜身份，猜出来的东西还被 T4 原样存库、印进群发邮件。顺带一提，那句话里「说话人」和代称之间没有空格，代称是中文时才读得通。

契约：

- 任意形如 `SPEAKER_<数字>` 的说话人都要有可读中文代称，超出现有五项时按规则生成，不许把裸 id 交给模型。
- 现有 `SPEAKER_00..04` → `甲乙丙丁戊` 这五项映射保持现状（preserve）；`SPEAKER_NAMES` 这个模块级名字保留。

> 超出五项怎么生成：继续按天干往下排（己庚辛壬癸，排完十个之后怎么办）、用「说话人6」这种直白写法、还是别的一致的规则——你定，写清规则和它的上限行为。另外，不像 `SPEAKER_\d+` 的键（人工补录进来的 `"法师A"`、`"监院"`）怎么办，也一并写明。
> 选定方案：（待填）

**T8 离线 import 必须成功，且不发任何网络请求。** 现状模块末尾 `ai_service = AIService()`，`AIService.__init__` 里就构造 `AsyncOpenAI(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL)`，而 `OPENAI_API_KEY` 默认是空串。

契约：

- 在没有 `OPENAI_API_KEY`、完全离线的机器上，`import backend.services.ai_service` 必须成功、不抛异常、不发出任何网络请求。
- 模块级单例名 `ai_service` 保留，类型仍是 `AIService`；`ai_service.client` 这个属性名保留（测试要顶掉它）。
- 构造 `AsyncOpenAI` 本身不许在 import 阶段联网，也不许因为 key 为空就抛异常。
- 这条现状是好的，属 preserve：要有测试钉住，别在改前面几条的时候把它弄坏（比如把 `AIService()` 挪到模块顶层做别的事、或者在 `__init__` 里加一次探活请求）。

---

## 测试要求

- **不许联网，不许发出真实请求。** 用标准库写一个替身客户端，`svc = AIService()` 之后 `svc.client = FakeClient(...)` 顶掉。替身要能：按调用返回预设正文、按调用抛预设异常、记录每一次调用的 `model` / `messages` / `temperature`、统计调用次数。`transcribe_audio` 走的是 `self.client.audio.transcriptions.create`，替身要把这条路径也顶上。
- pyannote 用 `sys.modules` 塞一个假模块，或者 monkeypatch 掉 `diarize_speakers` 里那句 import 能取到的东西。不许装它。
- 不要设置 `OPENAI_API_KEY`；不要依赖仓库里或机器上的 `.env`；临时文件一律用 `tmp_path`。
- `pytest.ini` 里已经开了 `asyncio_mode = auto`，`async def test_*` 直接写就行，不要加 `@pytest.mark.asyncio`，也不要改 `pytest.ini`。
- T1–T8 每一条至少一条测试；其中 T1、T4、T6 每条至少两条（成功路径 + 失败路径各一条）。
- 断言要落在可观察的产出上：返回的字典的键集合与值、返回列表的长度和每一项、实际发出的 `messages` 文本里含不含某个词、调用次数、`caplog` 里那条 warning 的内容、抛出的异常类型和它的可读原因——不要只断言「没抛异常」或者「返回了一个字符串」。
- **至少两条负例**：一条钉住 T1 的凭据脱敏（把脱敏那一步去掉、改回 `str(e)` 直接拼进返回值或日志，这条测试必须变红）；一条钉住 T4 的白名单归一（把归一化那一步去掉、改回模型原文照存，这条测试必须变红）。
- 跑完整套测试之后仓库工作树必须干净：不留 `output/`、不留 `temp_audio/`、不留 `.db`、不留 `__pycache__` 以外的新文件，不改 `README.md` 以外的既有文件。

## 已定口径与自定口径

上面 T1、T2、T3、T4、T6、T7 六处标了「选定一种并写明」，是本次要你拿主意的地方。选完把选择和理由填进对应条目下面那个「选定方案：（待填）」占位处，并让实现与之一致。六处口径彼此要自洽：T1 与 T2 的失败表达方式应当同一种口味，T3 与 T6 的「跳过还是抛」要想清楚谁在替运维兜底。评审只看 README 里这六处和你最后的总结。

## 本次不要求处理（另有排期，别顺手改）

下面这些确实是问题，但本次不验收，也不要动。发现了写进总结就行：

- `" ".join(texts)` 把中文句子用空格拼起来，凭空多出空格；`combined_text[:2000]` 硬截断，长会后半段发言根本没进提示词，模型是在半份材料上认身份的。
- 提示词里那段角色说明与 T4 的白名单是两处独立维护的文本，加一个角色要改两个地方，迟早对不上。
- `transcribe_audio` 里 `with open(...)` 握着文件句柄跨了整个 `await`，大文件加慢网络会长时间占着句柄。
- 三个 `generate_*` 的模型名与 `temperature`（`gpt-4o`/0.7、`gpt-4o`/0.8、`gpt-4o`/0.6、`gpt-4o-mini`/0.3）硬编码在函数体里，没有配置项——**这些值本次保持现状**，别顺手抽成配置，也别顺手调。
- 上游整条处理链包在一个 `try/except Exception` 里，异常只 `print` 一行，没有 `logging`，也没把原因写回 `meeting`。
- `align_transcript_with_speakers` 的平局与切分算法（见 T6 后面那段「不在范围」）。

### 本次做题人填写

（把你的取舍和理由写在这里。）