import { useState } from "react";
import "./App.css";
import EvolvingEnvironment from "./EvolvingEnvironment";
import MultiAgentLocalization from "./MultiAgentLocalization";

const TABS = [
  { id: "localization", label: "Multi-Agent Localization" },
  { id: "evolving", label: "Evolving Environment" },
];

export default function App() {
  const [activeTab, setActiveTab] = useState(TABS[0].id);

  return (
    <div className="app-root">
      <nav className="top-nav">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            className={`nav-tab ${activeTab === tab.id ? "active" : ""}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>
      <div className="app-container">
        {activeTab === "evolving" && <EvolvingEnvironment />}
        {activeTab === "localization" && <MultiAgentLocalization />}
      </div>
    </div>
  );
}
