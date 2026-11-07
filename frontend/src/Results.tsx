import { useState } from 'react';
import { MetricCard } from './components';
import type { ScreeningResult } from './types';

const tabs = ['RAG 证据', 'Agent 轨迹', '专病复核', '结构化数据'];
const num = (value: number) => value.toFixed(2);

export function Results({
  result,
  preview,
  busy,
  error,
}: {
  result: ScreeningResult | null;
  preview: string;
  busy: boolean;
  error: string;
}) {
  const [tab, setTab] = useState(0);
  return (
    <section className="results" aria-labelledby="results-title">
      <div className="panel" aria-hidden="true" />
      <h2 className="section-title" id="results-title">
        筛查结果
      </h2>
      {error && (
        <div className="error" role="alert">
          {error}
        </div>
      )}
      {busy ? (
        <div className="loading" role="status">
          <span className="spinner" />
          正在执行 LangGraph 多智能体筛查流程...
        </div>
      ) : !result ? (
        <>
          <div className="result-banner" style={{ background: '#f0fdf4' }}>
            <div className="result-label">Waiting For Input</div>
            <div className="result-main" style={{ color: '#15803d' }}>
              请在左侧输入症状并上传胸片
            </div>
          </div>
          <p>系统将展示筛查结论、三类风险分数、RAG 召回证据、Agent 执行轨迹和下一步检查建议。</p>
        </>
      ) : (
        <>
          <div className="result-banner" style={{ background: result.tone.background }}>
            <div className="result-label">Screening Result · {result.tone.tag}</div>
            <div className="result-main" style={{ color: result.tone.color }}>
              {result.screening_result}
            </div>
          </div>
          {preview && (
            <figure className="image-preview">
              <img src={preview} alt="上传的胸部影像" />
              <figcaption>上传的胸部影像</figcaption>
            </figure>
          )}
          <div className="three-columns">
            <MetricCard
              title="急性感染方向"
              value={num(result.image_scores['感染性异常'] ?? 0)}
              note="肺炎/急性感染性病变风险"
              accent="#d97706"
            />
            <MetricCard
              title="结核慢性方向"
              value={num(result.image_scores['慢性异常'] ?? 0)}
              note="结核样/慢性异常风险"
              accent="#0f766e"
            />
            <MetricCard
              title="综合置信度"
              value={num(result.confidence)}
              note="多源证据综合置信度"
              accent="#16a34a"
            />
          </div>
          <h4>结构化问诊</h4>
          <div className="info">{result.enriched_symptom}</div>
          <div className="two-columns">
            <div>
              <h4>主要依据</h4>
              <p>{result.rationale || '系统未返回主要依据。'}</p>
              <h4>鉴别诊断</h4>
              <p>{result.differential || '系统未返回鉴别诊断。'}</p>
            </div>
            <div>
              <h4>建议检查</h4>
              <p>{result.check_suggestion || '系统未返回检查建议。'}</p>
              <h4>复核意见</h4>
              <p>{result.review_note || '系统未返回复核意见。'}</p>
            </div>
          </div>
          <h4>模型与路由</h4>
          <div className="three-columns route-metrics">
            {[
              ['运行模式', result.used_llm ? 'LLM增强' : '本地Agent'],
              ['路由分支', result.route_decision],
              ['自反馈轮次', String(result.iteration)],
            ].map(([label, value]) => (
              <div key={label}>
                <div>{label}</div>
                <strong>{value}</strong>
              </div>
            ))}
          </div>
          <p className="caption">{result.route_reason}</p>
          <div className="tabs" role="tablist" aria-label="筛查详情">
            {tabs.map((text, index) => (
              <button
                key={text}
                id={`tab-${index}`}
                role="tab"
                aria-selected={tab === index}
                aria-controls={`tab-panel-${index}`}
                tabIndex={tab === index ? 0 : -1}
                onClick={() => setTab(index)}
                onKeyDown={(event) => {
                  if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
                    const next = (tab + (event.key === 'ArrowRight' ? 1 : 3)) % 4;
                    setTab(next);
                    document.getElementById(`tab-${next}`)?.focus();
                  }
                }}
              >
                {text}
              </button>
            ))}
          </div>
          <div
            className="tab-content"
            role="tabpanel"
            id={`tab-panel-${tab}`}
            aria-labelledby={`tab-${tab}`}
          >
            {tab === 0 && (
              <>
                <p>RAG感染性证据：{num(result.rag_infectious_score)}</p>
                <p>RAG慢性异常证据：{num(result.rag_chronic_score)}</p>
                <p>{result.rag_route_note}</p>
                {result.retrieved_knowledge.map((item, i) => (
                  <p key={i}>
                    <b>{i + 1}.</b> {item}
                  </p>
                ))}
              </>
            )}
            {tab === 1 &&
              result.agent_trace.map((item, i) => (
                <p key={i}>
                  <b>Step {i + 1}</b>　{item}
                </p>
              ))}
            {tab === 2 && (
              <div className="two-columns">
                <div>
                  <MetricCard
                    title="肺炎专病复核"
                    value={num(result.infectious_specialist_score)}
                    note={result.infectious_specialist_source}
                    accent="#d97706"
                  />
                  <p>{result.infectious_specialist_note}</p>
                </div>
                <div>
                  <MetricCard
                    title="结核/慢性复核"
                    value={num(result.chronic_specialist_score)}
                    note={result.chronic_specialist_source}
                    accent="#0f766e"
                  />
                  <p>{result.chronic_specialist_note}</p>
                </div>
              </div>
            )}
            {tab === 3 && (
              <pre aria-label="结构化数据 JSON">
                {JSON.stringify(result.structured_data, null, 2)}
              </pre>
            )}
          </div>
        </>
      )}
    </section>
  );
}
