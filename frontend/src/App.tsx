import { useEffect, useRef, useState } from 'react';
import type { ChangeEvent, FormEvent } from 'react';
import { Hero, Pills, Sidebar, Workflow } from './components';
import { Results } from './Results';
import type { Answers, ScreeningResult } from './types';

export default function App() {
  const [answers, setAnswers] = useState<Answers>({
    symptom: '',
    duration_value: '未说明',
    fever_value: '未说明',
    sputum_value: '未说明',
    sputum_color: '未说明',
    chest_value: '未说明',
    tb_value: '未说明',
  });
  const [duration, setDuration] = useState('');
  const [sputum, setSputum] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [fileError, setFileError] = useState('');
  const [preview, setPreview] = useState('');
  const [resultPreview, setResultPreview] = useState('');
  const [result, setResult] = useState<ScreeningResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [dragging, setDragging] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const picker = useRef<HTMLInputElement>(null);
  const controller = useRef<AbortController | null>(null);
  useEffect(() => () => controller.current?.abort(), []);
  useEffect(() => {
    if (!file) {
      setPreview('');
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  useEffect(
    () => () => {
      if (resultPreview) URL.revokeObjectURL(resultPreview);
    },
    [resultPreview],
  );
  const clearResult = () => {
    setResult(null);
    setResultPreview('');
    setError('');
  };
  const update = (name: keyof Answers, value: string) => {
    clearResult();
    setAnswers((current) => ({ ...current, [name]: value }));
  };
  function chooseFile(selected: File | null) {
    if (!selected) return;
    if (!/\.(png|jpe?g)$/i.test(selected.name)) {
      setFileError('请选择 PNG、JPG 或 JPEG 影像。');
      return;
    }
    if (selected.size > 200 * 1024 * 1024) {
      setFileError('影像文件不能超过200MB。');
      return;
    }
    clearResult();
    setFile(selected);
    setFileError('');
  }
  function fileChanged(event: ChangeEvent<HTMLInputElement>) {
    chooseFile(event.target.files?.[0] ?? null);
    event.target.value = '';
  }
  async function submit(event: FormEvent) {
    event.preventDefault();
    if (busy) return;
    const payload = {
      ...answers,
      duration_value: answers.duration_value === '自定义' ? duration : answers.duration_value,
      sputum_color: answers.sputum_color === '自定义' ? sputum : answers.sputum_color,
    };
    const form = new FormData();
    form.append('answers', JSON.stringify(payload));
    if (file) form.append('image', file);
    clearResult();
    setBusy(true);
    controller.current = new AbortController();
    try {
      const response = await fetch('/api/screen', {
        method: 'POST',
        body: form,
        signal: controller.current.signal,
      });
      const data = await response.json();
      if (!response.ok)
        throw new Error(
          typeof data.detail === 'string' ? data.detail : '筛查未完成，请检查配置与输入后重试。',
        );
      setResult(data);
      setResultPreview(file ? URL.createObjectURL(file) : '');
    } catch (cause) {
      if (!(cause instanceof DOMException && cause.name === 'AbortError'))
        setError(cause instanceof Error ? cause.message : '筛查未完成，请检查配置与输入后重试。');
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className={sidebarOpen ? 'app' : 'app sidebar-closed'}>
      <div className="app-bar">
        <button
          className="sidebar-toggle"
          type="button"
          aria-label={sidebarOpen ? '收起侧栏' : '展开侧栏'}
          aria-expanded={sidebarOpen}
          onClick={() => setSidebarOpen(!sidebarOpen)}
        >
          {sidebarOpen ? '‹' : '›'}
        </button>
      </div>
      {sidebarOpen && <Sidebar />}
      <main className="block-container">
        <Hero />
        <div className="screening-columns">
          <div className="input-column">
            <div className="panel" aria-hidden="true" />
            <h2 className="section-title">病例输入</h2>
            <div className="subtle">
              先用快速问诊完成结构化选择，再补充患者原话，适合现场演示和基层问诊。
            </div>
            <div className="question-box">
              <div className="question-box-title">快速结构化问诊</div>
              <div className="question-box-subtitle">
                先选择关键症状，系统会自动组合为多智能体推理输入；必要时可选择“自定义”。
              </div>
            </div>
            <form onSubmit={submit}>
              <fieldset disabled={busy}>
                <Pills
                  label="病程持续多久？"
                  options={['未说明', '1天', '2天', '3天', '1周', '2周', '1个月', '更久', '自定义']}
                  value={answers.duration_value}
                  onChange={(value) => update('duration_value', value)}
                />
                {answers.duration_value === '自定义' && (
                  <label className="custom-field">
                    请输入病程
                    <input
                      value={duration}
                      onChange={(e) => {
                        clearResult();
                        setDuration(e.target.value);
                      }}
                      placeholder="例如：5天、半个月、2个月"
                    />
                  </label>
                )}
                <div className="two-columns">
                  <div>
                    <Pills
                      label="是否发热？"
                      options={['未说明', '有', '无']}
                      value={answers.fever_value}
                      onChange={(value) => update('fever_value', value)}
                    />
                    <Pills
                      label="是否咳痰？"
                      options={['未说明', '有', '无']}
                      value={answers.sputum_value}
                      onChange={(value) => update('sputum_value', value)}
                    />
                  </div>
                  <div>
                    <Pills
                      label="是否胸痛/气促？"
                      options={['未说明', '有', '无']}
                      value={answers.chest_value}
                      onChange={(value) => update('chest_value', value)}
                    />
                    <Pills
                      label="是否盗汗/消瘦/咯血？"
                      options={['未说明', '有', '无']}
                      value={answers.tb_value}
                      onChange={(value) => update('tb_value', value)}
                    />
                  </div>
                </div>
                <Pills
                  label="痰液特点"
                  options={['未说明', '白痰', '黄痰', '血痰', '自定义']}
                  value={answers.sputum_color}
                  onChange={(value) => update('sputum_color', value)}
                />
                {answers.sputum_color === '自定义' && (
                  <label className="custom-field">
                    请输入痰液特点
                    <input
                      value={sputum}
                      onChange={(e) => {
                        clearResult();
                        setSputum(e.target.value);
                      }}
                      placeholder="例如：少量黏痰、铁锈色痰、无明显痰"
                    />
                  </label>
                )}
                <label className="field-label" htmlFor="symptom">
                  补充症状描述（可选）
                  <span
                    className="help"
                    title="可以输入患者原话，例如：前几天嗓子痛，后来咳嗽、黄痰、发烧。"
                    tabIndex={0}
                    aria-label="症状输入帮助"
                  >
                    ?
                  </span>
                </label>
                <textarea
                  id="symptom"
                  value={answers.symptom}
                  onChange={(e) => update('symptom', e.target.value)}
                  placeholder="例如：发热3天，咳嗽咳痰，伴胸痛"
                  rows={4}
                />
                <div className="field-label" id="upload-label">
                  上传胸部影像
                </div>
                <div
                  className={`upload-zone ${dragging ? 'dragging' : ''}`}
                  onDragOver={(e) => {
                    e.preventDefault();
                    if (!busy) setDragging(true);
                  }}
                  onDragLeave={() => setDragging(false)}
                  onDrop={(e) => {
                    e.preventDefault();
                    setDragging(false);
                    if (!busy) chooseFile(e.dataTransfer.files[0] ?? null);
                  }}
                >
                  <input
                    ref={picker}
                    type="file"
                    accept=".png,.jpg,.jpeg"
                    onChange={fileChanged}
                    aria-labelledby="upload-label"
                  />
                  <span className="upload-icon" aria-hidden="true">
                    ↥
                  </span>
                  <div>
                    <span>Upload</span>
                    <small>200MB per file · PNG, JPG</small>
                  </div>
                  <button
                    type="button"
                    className="upload-button"
                    onClick={() => picker.current?.click()}
                  >
                    选择文件
                  </button>
                </div>
                {file && (
                  <div className="selected-file">
                    <span>{file.name}</span>
                    <button
                      type="button"
                      aria-label="移除已选影像"
                      onClick={() => {
                        clearResult();
                        setFile(null);
                        setFileError('');
                      }}
                    >
                      ×
                    </button>
                  </div>
                )}
                {fileError && (
                  <p className="error" role="alert">
                    {fileError}
                  </p>
                )}
                <p className="caption privacy">
                  请仅使用脱敏且已获授权的演示资料。影像在本地推理；若启用云端大模型，症状及派生证据将发送至所配置的服务。处理完成后删除本次上传的临时文件。
                </p>
                <button type="submit" className="submit-button">
                  开始辅助筛查
                </button>
              </fieldset>
            </form>
            <Workflow />
          </div>
          <Results
            result={result}
            preview={resultPreview || (!result ? preview : '')}
            busy={busy}
            error={error}
          />
        </div>
      </main>
    </div>
  );
}
