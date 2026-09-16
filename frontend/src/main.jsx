import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import axios from "axios";
import "./styles/app.css";

const API = "http://127.0.0.1:8000";

/* =========================================================
   SHARED MARKS / ICONS
   ========================================================= */

function SealMark({ size = 42 }) {
  return (
    <div className="seal-mark" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox="0 0 42 42" fill="none">
        <circle cx="21" cy="21" r="19.5" stroke="#a9812f" strokeWidth="1.2" />
        <circle cx="21" cy="21" r="15.5" stroke="#a9812f" strokeWidth="0.6" opacity="0.6" />
        <path
          d="M21 11 L21 31 M14 15 L28 15 M13 15 L13 18 L17 22 L14 26 L20 26 M29 15 L29 18 L25 22 L28 26 L22 26"
          stroke="#a9812f"
          strokeWidth="1.1"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </div>
  );
}

function DocumentIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
      <path
        d="M7 3h7l5 5v13a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1Z"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
      <path d="M14 3v5h5" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
      <path d="M9 13h6M9 17h6M9 9h2" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  );
}

function PrecedentIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
      <path
        d="M6 3h12a1 1 0 0 1 1 1v16a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1Z"
        stroke="currentColor"
        strokeWidth="1.5"
      />
      <path d="M8 7h8M8 11h8M8 15h5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  );
}

/* Normalizes whatever relevance label the backend sends into one of
   three known visual tiers, instead of requiring an exact string match. */
function relevanceTier(label) {
  const text = String(label || "").toLowerCase();

  if (/(high|strong)/.test(text)) return "relevance-high";
  if (/(medium|moderate)/.test(text)) return "relevance-medium";
  if (/(low|weak|potential)/.test(text)) return "relevance-low";

  return "";
}

/* =========================================================
   PRECEDENT CARD
   ========================================================= */

function PrecedentCard({ precedent }) {
  const relevance =
    precedent.relevance ||
    precedent.relevance_level ||
    precedent.precedent_strength ||
    "Potentially Relevant";

  const legalPrinciple =
    precedent.legal_principle || precedent.principle || precedent.legal_principles || "";

  const whyRelevant = precedent.why_relevant || precedent.relevance_reason || precedent.reason || "";

  const similarity = precedent.similarity !== undefined ? precedent.similarity : null;

  return (
    <article className="precedent-card">
      <div className="precedent-card-header">
        <div className="precedent-title-area">
          <div className="precedent-icon">
            <PrecedentIcon />
          </div>

          <div>
            <h3>{precedent.title || "Potentially Relevant Precedent"}</h3>

            <div className="precedent-meta">
              {precedent.case_id && <span>Case ID: {precedent.case_id}</span>}
              {precedent.year && <span>Year: {precedent.year}</span>}
            </div>
          </div>
        </div>

        <div className={`relevance-badge ${relevanceTier(relevance)}`}>{relevance}</div>
      </div>

      {similarity !== null && (
        <div className="precedent-similarity">
          <span>Similarity with uploaded case</span>
          <strong>{similarity}%</strong>
        </div>
      )}

      {legalPrinciple && (
        <div className="precedent-block principle-block">
          <h4>Legal Principle</h4>
          <p>{legalPrinciple}</p>
        </div>
      )}

      {whyRelevant && (
        <div className="precedent-block">
          <h4>Why May This Precedent Be Relevant?</h4>
          <p>{whyRelevant}</p>
        </div>
      )}

      {precedent.case_summary && (
        <div className="precedent-block">
          <h4>Previous Case Summary</h4>
          <p>{precedent.case_summary}</p>
        </div>
      )}

      {precedent.summary && !precedent.case_summary && (
        <div className="precedent-block">
          <h4>Previous Case Summary</h4>
          <p>{precedent.summary}</p>
        </div>
      )}

      {precedent.judgment && (
        <div className="precedent-block">
          <h4>Judgment Context</h4>
          <p>{precedent.judgment}</p>
        </div>
      )}

      <div className="precedent-disclaimer">
        This system identifies potentially relevant precedents for judicial research.
        Final applicability and legal interpretation must be determined by the
        appropriate judicial authority.
      </div>
    </article>
  );
}

/* =========================================================
   PRECEDENT ANALYSIS SECTION
   ========================================================= */

function PrecedentAnalysis({ result }) {
  const analysis = result.precedent_analysis || result.precedents || null;

  if (!analysis) {
    return null;
  }

  let precedents = [];
  let overallSummary = "";

  if (Array.isArray(analysis)) {
    precedents = analysis;
  } else {
    precedents = analysis.precedents || analysis.results || analysis.cases || [];
    overallSummary = analysis.summary || analysis.overview || "";
  }

  if (!Array.isArray(precedents)) {
    precedents = [];
  }

  return (
    <section className="precedent-section">
      <div className="section-label">Precedent Analysis</div>

      <div className="precedent-heading-row">
        <div>
          <h2>Previous judgments with potentially relevant legal principles.</h2>

          <p className="section-description">
            Similar cases are further analysed to identify judgments that may contain
            legal principles relevant to the uploaded case.
          </p>
        </div>

        <div className="precedent-status">
          <span className="status-dot" />
          Research support
        </div>
      </div>

      {overallSummary && (
        <div className="precedent-overview">
          <h3>Precedent Analysis Overview</h3>
          <p>{overallSummary}</p>
        </div>
      )}

      {precedents.length === 0 ? (
        <div className="empty precedent-empty">
          No potentially relevant precedents were identified from the retrieved cases.
        </div>
      ) : (
        <div className="precedent-list">
          {precedents.map((precedent, index) => (
            <PrecedentCard key={precedent.case_id || precedent.id || index} precedent={precedent} />
          ))}
        </div>
      )}
    </section>
  );
}

/* =========================================================
   JUDICIARY CASE RESULTS
   ========================================================= */

function JudiciaryCase({ result }) {
  const explanation = result.explanation || {};
  const similarCases = result.results || [];

  return (
    <div className="results">
      <section className="case-overview">
        <div className="section-label">Case Overview</div>

        <h2>Classification of the uploaded case</h2>

        <div className="classification">
          <div>
            <span>Case Type</span>
            <strong>{result.case_type || "Not available"}</strong>
          </div>

          <div>
            <span>Sub-case</span>
            <strong>{result.sub_case_type || "Not available"}</strong>
          </div>
        </div>
      </section>

      <section className="explanation-card">
        <div className="section-label">Case Explanation</div>

        <h2>A simplified explanation of the judgment</h2>

        {explanation.summary && (
          <div className="explanation-block">
            <h3>Case Summary</h3>
            <p>{explanation.summary}</p>
          </div>
        )}

        {explanation.what_happened && (
          <div className="explanation-block">
            <h3>What Happened</h3>
            <p>{explanation.what_happened}</p>
          </div>
        )}

        {explanation.evidence_considered && (
          <div className="explanation-block">
            <h3>What Evidence Was Considered</h3>
            <p>{explanation.evidence_considered}</p>
          </div>
        )}

        {explanation.legal_issue && (
          <div className="explanation-block">
            <h3>What Was the Main Question?</h3>
            <p>{explanation.legal_issue}</p>
          </div>
        )}

        {explanation.court_decision && (
          <div className="explanation-block decision">
            <h3>What Did the Court Decide?</h3>
            <p>{explanation.court_decision}</p>
          </div>
        )}

        <div className="verification-note">
          This explanation is generated from the uploaded judgment. The original
          judgment should be checked for complete details.
        </div>
      </section>

      <section className="similar-section">
        <div className="section-label">Similar Cases</div>

        <h2>Previous cases identified as similar to the uploaded judgment.</h2>

        {similarCases.length === 0 ? (
          <div className="empty">No similar cases were found.</div>
        ) : (
          <div className="similar-list">
            {similarCases.map((item) => (
              <article className="similar-case" key={item.case_id}>
                <div className="case-header">
                  <div>
                    <h3>{item.title}</h3>

                    <div className="case-meta">
                      <span>Case ID: {item.case_id}</span>
                      <span>{item.case_type}</span>
                      <span>{item.sub_case_type}</span>
                      {item.year && <span>Year: {item.year}</span>}
                    </div>
                  </div>

                  <div className="similarity">
                    {item.similarity}%<small>Similarity</small>
                  </div>
                </div>

                <div className="case-information">
                  <div>
                    <h4>Case Summary</h4>
                    <p>{item.summary || "No case summary is available."}</p>
                  </div>

                  <div>
                    <h4>Case Details</h4>
                    <p>{item.facts || "No case details are available."}</p>
                  </div>

                  <div>
                    <h4>Judgment</h4>
                    <p>{item.judgment || "No judgment information is available."}</p>
                  </div>
                </div>
              </article>
            ))}
          </div>
        )}
      </section>

      <PrecedentAnalysis result={result} />
    </div>
  );
}

/* =========================================================
   CITIZEN CASE RESULTS
   ========================================================= */

function CitizenCase({ result }) {
  const explanation = result.explanation || {};

  return (
    <div className="results">
      <section className="citizen-case">
        <div className="section-label">Your Case</div>

        <h2>Simple explanation of your case</h2>

        {explanation.summary && (
          <div className="citizen-block">
            <h3>What is this case about?</h3>
            <p>{explanation.summary}</p>
          </div>
        )}

        {explanation.what_happened && (
          <div className="citizen-block">
            <h3>What happened?</h3>
            <p>{explanation.what_happened}</p>
          </div>
        )}

        {explanation.court_decision && (
          <div className="citizen-decision">
            <h3>What did the court decide?</h3>
            <p>{explanation.court_decision}</p>
          </div>
        )}

        <div className="citizen-note">
          This explanation is provided in simple language so that the main events
          and court decision can be understood without requiring legal knowledge.
        </div>
      </section>
    </div>
  );
}

/* =========================================================
   AUTH SHELL
   ========================================================= */

function AuthShell({ eyebrow, headline, points, children }) {
  return (
    <main className="login-page">
      <section className="auth-panel">
        <div className="auth-panel-top">
          <SealMark size={44} />

          <div className="login-eyebrow" style={{ marginTop: 22 }}>
            {eyebrow}
          </div>

          <h2>{headline}</h2>

          <ul className="auth-points">
            {points.map((point, i) => (
              <li key={point}>
                <span className="point-index">{String(i + 1).padStart(2, "0")}</span>
                <p>{point}</p>
              </li>
            ))}
          </ul>
        </div>

        <div className="auth-panel-bottom">LEGAL AI — JUDICIARY CASE INTELLIGENCE</div>
      </section>

      <div className="auth-form-side">{children}</div>
    </main>
  );
}

const AUTH_POINTS = [
  "Extract named legal entities, courts, statutes and parties from an uploaded judgment.",
  "Retrieve the most similar previous cases using category-aware semantic search.",
  "Identify potentially relevant precedents and legal principles for judicial research."
];

/* =========================================================
   LOGIN PAGE
   ========================================================= */

function Login({ onLogin, onRegister }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function submit(event) {
    event.preventDefault();

    setError("");
    setLoading(true);

    try {
      const response = await axios.post(`${API}/auth/login`, {
        email,
        password
      });

      const token = response.data.access_token;
      const role = response.data.role;

      localStorage.setItem("token", token);
      localStorage.setItem("role", role);

      onLogin(token, role);
    } catch (error) {
      setError(
        error.response?.data?.detail || "Login failed. Please check your credentials."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <AuthShell
      eyebrow="Judiciary Case Intelligence"
      headline="Structured research for every judgment you review."
      points={AUTH_POINTS}
    >
      <div className="login-card">
        <div className="login-eyebrow">Welcome back</div>
        <h1>Sign in</h1>
        <p className="login-subtitle">Continue to your workspace.</p>

        <form onSubmit={submit}>
          <label>Email</label>
          <input
            type="email"
            placeholder="Enter your email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />

          <label>Password</label>
          <input
            type="password"
            placeholder="Enter your password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />

          <button className="primary-button" disabled={loading}>
            {loading && <span className="spinner" />}
            {loading ? "Signing in…" : "Sign in"}
          </button>
        </form>

        {error && <div className="error">{error}</div>}

        <div className="auth-switch">
          <span>Don't have an account?</span>
          <button type="button" className="text-button" onClick={onRegister}>
            Create an account
          </button>
        </div>
      </div>
    </AuthShell>
  );
}

/* =========================================================
   REGISTRATION PAGE
   ========================================================= */

function Register({ onRegistered, onBackToLogin }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [role, setRole] = useState("citizen");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [loading, setLoading] = useState(false);

  async function submit(event) {
    event.preventDefault();

    setError("");
    setSuccess("");

    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    if (password.length < 6) {
      setError("Password must contain at least 6 characters.");
      return;
    }

    setLoading(true);

    try {
      const response = await axios.post(`${API}/auth/register`, {
        email,
        password,
        role
      });

      setSuccess(response.data.message || "Registered successfully.");

      setEmail("");
      setPassword("");
      setConfirmPassword("");
      setRole("citizen");

      setTimeout(() => {
        onRegistered();
      }, 1200);
    } catch (error) {
      setError(error.response?.data?.detail || "Registration failed. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <AuthShell
      eyebrow="Judiciary Case Intelligence"
      headline="One workspace for judiciary research and citizen understanding."
      points={AUTH_POINTS}
    >
      <div className="login-card">
        <div className="login-eyebrow">Get started</div>
        <h1>Create account</h1>
        <p className="login-subtitle">Set up your Legal AI account.</p>

        <form onSubmit={submit}>
          <label>Email</label>
          <input
            type="email"
            placeholder="Enter your email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />

          <label>Password</label>
          <input
            type="password"
            placeholder="Create a password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />

          <label>Confirm Password</label>
          <input
            type="password"
            placeholder="Confirm your password"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            required
          />

          <label>Account Type</label>
          <div className="segmented" role="group" aria-label="Account type">
            <button
              type="button"
              className="segmented-option"
              aria-pressed={role === "citizen"}
              onClick={() => setRole("citizen")}
            >
              Citizen
            </button>
            <button
              type="button"
              className="segmented-option"
              aria-pressed={role === "judiciary"}
              onClick={() => setRole("judiciary")}
            >
              Judiciary
            </button>
          </div>
          <p className="field-hint">
            {role === "judiciary"
              ? "Judiciary accounts see detected entities, categorization, similar cases and precedent analysis."
              : "Citizen accounts see a plain-language explanation of the uploaded case."}
          </p>

          <button className="primary-button" disabled={loading}>
            {loading && <span className="spinner" />}
            {loading ? "Creating account…" : "Create Account"}
          </button>
        </form>

        {error && <div className="error">{error}</div>}
        {success && <div className="success-message">{success}</div>}

        <div className="auth-switch">
          <span>Already have an account?</span>
          <button type="button" className="text-button" onClick={onBackToLogin}>
            Sign in
          </button>
        </div>
      </div>
    </AuthShell>
  );
}

/* =========================================================
   MAIN APP
   ========================================================= */

function App() {
  const [token, setToken] = useState(localStorage.getItem("token"));
  const [role, setRole] = useState(localStorage.getItem("role") || "citizen");
  const [showRegister, setShowRegister] = useState(false);
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  function login(newToken, newRole) {
    setToken(newToken);
    setRole(newRole);
    setShowRegister(false);
  }

  function logout() {
    localStorage.removeItem("token");
    localStorage.removeItem("role");

    setToken(null);
    setRole("citizen");
    setResult(null);
    setFile(null);
    setError("");
  }

  async function upload() {
    if (!file) {
      setError("Please select a PDF first.");
      return;
    }

    setBusy(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(`${API}/upload`, formData, {
        headers: {
          Authorization: `Bearer ${token}`
        }
      });

      setResult(response.data);
    } catch (error) {
      setError(error.response?.data?.detail || "Analysis failed. Please try again.");
    } finally {
      setBusy(false);
    }
  }

  if (!token && showRegister) {
    return (
      <Register
        onRegistered={() => setShowRegister(false)}
        onBackToLogin={() => setShowRegister(false)}
      />
    );
  }

  if (!token) {
    return <Login onLogin={login} onRegister={() => setShowRegister(true)} />;
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <div>
          <div className="brand">
            <SealMark size={40} />
            <div>
              <strong>Legal AI</strong>
              <small>CASE INTELLIGENCE</small>
            </div>
          </div>

          <div className="workspace">
            {role === "judiciary" ? "Judiciary Workspace" : "Citizen Workspace"}
          </div>

          <div className="sidebar-meta">
            Signed in · {role === "judiciary" ? "Professional access" : "Citizen access"}
          </div>
        </div>

        <button className="logout-button" onClick={logout}>
          Logout
        </button>
      </aside>

      <main className="main-content">
        <header className="top-header">
          <div>
            <div className="section-label">
              {role === "judiciary" ? "Judiciary Portal" : "Citizen Portal"}
            </div>

            <h1>Case Intelligence</h1>

            <p>Upload a legal judgment or case document for analysis.</p>
          </div>

          <span className="role-pill">
            {role === "judiciary" ? "Judiciary role" : "Citizen role"}
          </span>
        </header>

        <section className="upload-card">
          <div className="upload-icon">
            <DocumentIcon />
          </div>

          <div className="upload-content">
            <h2>Upload Case Document</h2>
            <p>Select a PDF judgment or case document to begin the analysis.</p>

            <input
              id="pdf-upload"
              type="file"
              accept=".pdf"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />

            <label htmlFor="pdf-upload" className="file-label">
              {file ? file.name : "Choose PDF"}
            </label>
          </div>

          <button
            className="primary-button analyze-button"
            onClick={upload}
            disabled={!file || busy}
          >
            {busy && <span className="spinner" />}
            {busy ? "Analyzing…" : "Analyze Case"}
          </button>
        </section>

        {error && <div className="error page-error">{error}</div>}

        {result &&
          (role === "judiciary" ? (
            <JudiciaryCase result={result} />
          ) : (
            <CitizenCase result={result} />
          ))}
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
