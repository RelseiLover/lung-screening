import { useId } from 'react';

export function Pills({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: string[];
  value: string;
  onChange: (value: string) => void;
}) {
  const id = useId();
  return (
    <div className="question">
      <div id={id} className="field-label">
        {label}
      </div>
      <div className="pills" role="group" aria-labelledby={id}>
        {options.map((option) => (
          <button
            key={option}
            type="button"
            aria-pressed={value === option}
            onClick={() => onChange(value === option ? '' : option)}
          >
            {option}
          </button>
        ))}
      </div>
    </div>
  );
}

export function MetricCard({
  title,
  value,
  note = '',
  accent = '#45b786',
}: {
  title: string;
  value: string;
  note?: string;
  accent?: string;
}) {
  return (
    <div className="metric-card" style={{ borderTopColor: accent }}>
      <div className="metric-title">{title}</div>
      <div className="metric-value">{value}</div>
      <div className="metric-note">{note}</div>
    </div>
  );
}

export function Hero() {
  const capabilities = [
    ['胸片影像', 'EfficientNetV2-S 三分类基础模型'],
    ['专病复核', '肺炎 / 结核方向二次评估'],
    ['RAG 指南', 'ATS/IDSA 与 WHO 医学证据'],
    ['LangGraph', '条件分支与自反馈闭环'],
    ['DeepSeek', '诊断整合与报告解释'],
  ];
  return (
    <header className="hero">
      <h1>肺部异常辅助筛查与分诊系统</h1>
      <p>
        面向基层医疗场景的多模态智能体原型，融合胸片影像模型、症状问诊、RAG 医学指南、 LangGraph
        多智能体分支和 DeepSeek 诊断整合，输出可解释的肺部异常筛查结果。
      </p>
      <div className="capability-grid">
        {capabilities.map(([title, text]) => (
          <div className="capability-card" key={title}>
            <b>{title}</b>
            <span>{text}</span>
          </div>
        ))}
      </div>
    </header>
  );
}

export function Sidebar() {
  return (
    <aside className="sidebar">
      <h3>系统定位</h3>
      <p>肺部异常辅助筛查，不替代医生诊断。</p>
      <h3>核心能力</h3>
      <ul>
        {[
          '胸片三分类基础模型',
          '肺炎与结核专病复核',
          'RAG 指南证据召回',
          'DeepSeek 诊断整合',
          '自反馈与一致性校验',
        ].map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
      <hr />
      <h3>演示建议</h3>
      <ol>
        {[
          '输入症状或使用问诊补全',
          '上传胸部 X 光图像',
          '点击开始筛查',
          '展示 RAG 与 Agent 轨迹',
        ].map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ol>
    </aside>
  );
}

export function Workflow() {
  const nodes = [
    ['SymptomAgent', '抽取症状、病程和否定语境'],
    ['ImageAgent', '第一阶段影像分诊给出三类风险'],
    ['KnowledgeAgent', '召回肺炎与结核指南证据'],
    ['RouteAgent', '影像触发 + RAG 文本触发'],
    ['SpecialistAgents', '肺炎/结核专病模型复核'],
    ['DiagnosisAgent', 'DeepSeek 或本地规则综合诊断'],
    ['ReviewAgent', '图文冲突与一致性校验'],
    ['SelfFeedbackAgent', '低置信度时追加保守规则'],
    ['CheckAgent', '生成下一步检查建议'],
  ];
  return (
    <>
      <div className="panel" aria-hidden="true" />
      <h2 className="section-title">多智能体工作流</h2>
      <div className="timeline">
        {nodes.map(([title, text], index) => (
          <div className="node" data-step={String(index + 1).padStart(2, '0')} key={title}>
            <b>{title}</b>
            <span>{text}</span>
          </div>
        ))}
      </div>
    </>
  );
}
