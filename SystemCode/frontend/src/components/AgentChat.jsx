import { useEffect, useRef, useState } from "react";

const SUGGESTIONS = [
  "Why was this recording flagged?",
  "Which sensor changed most?",
  "What should I inspect next?",
];

export function RobotIcon({ size = 90 }) {
  return <img className="robot-render" src="/assets/robot-camera-3d.png" alt="" width={size} height={size} draggable="false" />;
}

function RobotWalker() {
  return (
    <span className="robot-walker" aria-hidden="true">
      <img className="robot-walker-piece robot-walker-body" src="/assets/robot-camera-3d.png" alt="" draggable="false" />
      <img className="robot-walker-piece robot-walker-left" src="/assets/robot-camera-3d.png" alt="" draggable="false" />
      <img className="robot-walker-piece robot-walker-middle" src="/assets/robot-camera-3d.png" alt="" draggable="false" />
      <img className="robot-walker-piece robot-walker-right" src="/assets/robot-camera-3d.png" alt="" draggable="false" />
    </span>
  );
}

export default function AgentChat({ caseId, open, onOpen, onClose }) {
  const [draft, setDraft] = useState("");
  const inputRef = useRef(null);
  const triggerRef = useRef(null);
  const close = () => {
    onClose();
    requestAnimationFrame(() => triggerRef.current?.focus());
  };

  useEffect(() => {
    if (!open) return;
    inputRef.current?.focus();
    const onKeyDown = (event) => {
      if (event.key === "Escape") close();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  return (
    <>
      {!open && <button ref={triggerRef} type="button" className="agent-launcher" onClick={onOpen}
        aria-label="Open MachineSense assistant" aria-expanded={false} aria-controls="agent-chat-panel" title="Open MachineSense assistant">
        <RobotWalker />
        <span className="agent-launcher-tip">Ask MachineSense</span>
      </button>}
      {open && <>
        <div className="agent-chat-shade" onClick={close} aria-hidden="true" />
        <section id="agent-chat-panel" className="agent-chat-panel" role="dialog" aria-modal="true" aria-label="MachineSense assistant">
          <header className="agent-chat-head">
            <RobotIcon size={44} />
            <div><strong>MachineSense assistant</strong><span>Investigation chat preview</span></div>
            <button type="button" onClick={close} className="agent-chat-close" aria-label="Close assistant">×</button>
          </header>
          <div className="agent-chat-body">
            <span className="agent-chat-state mono">AGENTIC ASSISTANT · NOT CONNECTED</span>
            <h2>Ask about the machine</h2>
            <p>{caseId ? "This diagnosis is ready to provide context when the investigation agents are connected." : "Start an investigation to give the assistant a recording to discuss."}</p>
            <p>The chat interface is a preview. Questions entered here are drafts and are not sent or answered yet.</p>
            <div className="agent-chat-suggestions" aria-label="Suggested questions">
              {SUGGESTIONS.map((question) => <button type="button" key={question} onClick={() => { setDraft(question); inputRef.current?.focus(); }}>{question}</button>)}
            </div>
          </div>
          <div className="agent-chat-compose">
            <label htmlFor="agent-chat-draft" className="mono kicker">Your question</label>
            <textarea id="agent-chat-draft" ref={inputRef} value={draft} onChange={(event) => setDraft(event.target.value)}
              placeholder="Ask a question about this recording…" rows={2} />
            <div className="agent-chat-actions"><span>Draft only · no message will be sent</span><button type="button" disabled title="The investigation agents are not connected">Send</button></div>
          </div>
        </section>
      </>}
    </>
  );
}
