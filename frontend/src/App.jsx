import { useState, useCallback } from "react";

// ─── NLP helpers (client-side, mirrors backend logic) ──────────────────────────

const SKILL_TAXONOMY = {
  programming: ["python","javascript","typescript","java","c++","c#","go","rust","ruby","scala","kotlin","swift","php","r","sql","bash","html","css","graphql"],
  frameworks: ["react","angular","vue","nextjs","django","flask","fastapi","spring","express","nodejs","tensorflow","pytorch","keras","scikit-learn","sklearn","xgboost","lightgbm","pandas","numpy","scipy","huggingface","langchain"],
  cloud_devops: ["aws","azure","gcp","docker","kubernetes","terraform","ansible","jenkins","kafka","rabbitmq","redis","nginx","linux","elasticsearch","prometheus","grafana"],
  databases: ["postgresql","mysql","sqlite","mongodb","cassandra","dynamodb","bigquery","snowflake","redshift","neo4j","pinecone","weaviate"],
  data_ai: ["machine learning","deep learning","nlp","computer vision","llm","rag","fine-tuning","data engineering","etl","spark","airflow","dbt","tableau","power bi","data pipeline","feature engineering","model deployment","mlops","a/b testing","statistics"],
  soft_skills: ["leadership","communication","teamwork","agile","scrum","project management","mentoring","stakeholder management","collaboration"]
};
const ALL_SKILLS = Object.values(SKILL_TAXONOMY).flat();

const STOPWORDS = new Set(["the","a","an","and","or","but","in","on","at","to","for","of","with","by","from","is","are","was","were","be","been","have","has","had","do","does","did","will","would","could","should","may","might","this","that","these","those","experience","work","working","looking","seeking","strong","good","excellent","ability","skills","knowledge","year","years","background","proven","must","required","preferred","using","use","used","plus","also","including"]);

function cleanText(t) { return t.toLowerCase().replace(/[^\w\s]/g," ").replace(/\s+/g," ").trim(); }
function tokenize(t) { return cleanText(t).split(" ").filter(w => w.length > 2 && !STOPWORDS.has(w)); }

function tfidf(text) {
  const tokens = tokenize(text);
  const freq = {};
  tokens.forEach(t => { freq[t] = (freq[t]||0) + 1; });
  const total = tokens.length || 1;
  const vec = {};
  Object.entries(freq).forEach(([w,c]) => { vec[w] = c/total; });
  return vec;
}

function cosineSim(a, b) {
  const common = Object.keys(a).filter(k => k in b);
  if (!common.length) return 0;
  const dot = common.reduce((s,k) => s + a[k]*b[k], 0);
  const magA = Math.sqrt(Object.values(a).reduce((s,v) => s+v*v, 0));
  const magB = Math.sqrt(Object.values(b).reduce((s,v) => s+v*v, 0));
  return magA && magB ? dot / (magA*magB) : 0;
}

function extractSkills(text) {
  const lower = text.toLowerCase();
  return [...new Set(ALL_SKILLS.filter(s => new RegExp(`\\b${s.replace(/[+#]/g,"\\$&")}\\b`).test(lower)))];
}

function matchResume(resumeText, jobText) {
  const vecA = tfidf(resumeText), vecB = tfidf(jobText);
  const rawSim = cosineSim(vecA, vecB);
  const matchScore = Math.min(Math.round(rawSim * 200 * 10) / 10, 100);

  const resumeSkills = extractSkills(resumeText);
  const jobSkills = extractSkills(jobText);
  const matched = jobSkills.filter(s => resumeSkills.includes(s));
  const missing = jobSkills.filter(s => !resumeSkills.includes(s));
  const bonus = resumeSkills.filter(s => !jobSkills.includes(s)).slice(0,10);

  const tokA = new Set(tokenize(resumeText));
  const tokB = new Set(tokenize(jobText));
  const sharedKeywords = [...tokA].filter(w => tokB.has(w) && !STOPWORDS.has(w)).slice(0,30);

  const coverage = matched.length / Math.max(matched.length + missing.length, 1);
  let hireability, hireReason;
  if (matchScore >= 75 && coverage >= 0.65) { hireability="High"; hireReason="Strong semantic alignment with extensive skill overlap."; }
  else if (matchScore >= 50 && coverage >= 0.40) { hireability="Medium"; hireReason="Moderate fit with some skill gaps; upskilling recommended."; }
  else { hireability="Low"; hireReason="Significant gaps in required skills and content alignment."; }

  let recommendation;
  if (matchScore >= 75) recommendation = `Excellent match! Your resume aligns strongly with this role. You have ${matched.length} of ${jobSkills.length} required skills. Focus on quantifying your achievements before applying.`;
  else if (matchScore >= 50) recommendation = `Good potential with room to improve. You match ${matched.length} of ${jobSkills.length} required skills. Consider adding experience with: ${missing.slice(0,3).join(", ")||"none identified"}.`;
  else recommendation = `Significant skill gaps detected. You meet ${matched.length} of ${jobSkills.length} required skills. Prioritize acquiring: ${missing.slice(0,4).join(", ")||"review the JD carefully"}.`;

  return { matchScore, hireability, hireReason, matched, missing, bonus, sharedKeywords, recommendation,
    stats: { resumeWords: resumeText.split(/\s+/).length, jdWords: jobText.split(/\s+/).length, resumeSkills: resumeSkills.length, jdSkills: jobSkills.length } };
}

// ─── AI-powered analysis via Claude API ────────────────────────────────────────
async function getAIAnalysis(resumeText, jobText, basicResult) {
  const prompt = `You are an expert career coach and HR professional. Analyze this resume vs job description match.

MATCH SCORE: ${basicResult.matchScore}/100
HIREABILITY: ${basicResult.hireability}
MATCHED SKILLS: ${basicResult.matched.join(", ")||"none"}
MISSING SKILLS: ${basicResult.missing.join(", ")||"none"}

RESUME (excerpt):
${resumeText.slice(0,800)}

JOB DESCRIPTION (excerpt):
${jobText.slice(0,800)}

Respond with ONLY a JSON object (no markdown, no backticks):
{
  "executive_summary": "2-sentence sharp, direct summary of fit",
  "strengths": ["strength 1","strength 2","strength 3"],
  "gaps": ["gap 1","gap 2","gap 3"],
  "action_items": ["specific action 1","specific action 2","specific action 3"],
  "resume_tips": ["tip 1","tip 2"],
  "interview_prep": "One key topic to prepare for"
}`;

  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1000,
      messages: [{ role: "user", content: prompt }]
    })
  });
  const data = await response.json();
  const text = data.content?.find(b => b.type==="text")?.text || "{}";
  try { return JSON.parse(text.replace(/```json|```/g,"").trim()); }
  catch { return null; }
}

// ─── Sample data ───────────────────────────────────────────────────────────────
const SAMPLE_RESUME = `SARAH CHEN – Senior Software Engineer
sarah.chen@email.com | github.com/sarahchen

SUMMARY
Results-driven software engineer with 6 years of experience building scalable web applications and data pipelines. Passionate about machine learning and cloud-native architectures. Strong leadership in cross-functional agile teams.

SKILLS
Python, JavaScript, TypeScript, SQL, React, FastAPI, Django, TensorFlow, scikit-learn, pandas, numpy, AWS, Docker, Kubernetes, GitHub Actions, Terraform, PostgreSQL, MongoDB, Redis, BigQuery, machine learning, deep learning, NLP, feature engineering, model deployment, MLOps, data pipeline, ETL, airflow, dbt, A/B testing, agile, scrum, mentoring

EXPERIENCE
Senior Software Engineer | DataFlow Inc. | 2022–Present
• Led real-time data pipeline processing 500K events/day (Python, Kafka, AWS)
• Built 3 ML models for churn prediction at 87% accuracy (scikit-learn, TensorFlow)
• Architected React + FastAPI platform serving 50K MAU
• Mentored 4 junior engineers; reduced delivery time 30% via agile practices
• CI/CD pipelines with GitHub Actions, Docker, Kubernetes

Software Engineer | Nexus Tech | 2019–2021
• Full-stack Python/Django + React features for e-commerce platform
• PostgreSQL schemas for 2M+ record systems
• ETL pipelines with Apache Airflow and dbt for BI dashboards
• A/B testing framework using statistics and Python

EDUCATION: B.S. Computer Science, UC Berkeley 2019
PROJECTS: Open-source NLP toolkit (500+ stars) using huggingface, transformers`;

const SAMPLE_JD = `Senior ML Engineer – AI Platform Team | TechCorp AI

We are looking for a Senior ML Engineer to design, build, and scale machine learning systems powering our core products.

REQUIRED SKILLS
• 5+ years software engineering, 2+ years ML engineering
• Strong Python programming; TypeScript/JavaScript a plus
• ML frameworks: TensorFlow, PyTorch, scikit-learn, XGBoost
• Cloud platforms: AWS, GCP, or Azure
• Docker, Kubernetes containerization
• SQL and data warehouses: BigQuery, Redshift, Snowflake
• Data pipelines: Apache Airflow, Spark, dbt
• NLP and LLMs preferred; MLOps practices
• Leadership, communication, agile/scrum experience

NICE TO HAVE
• LangChain, Hugging Face, RAG systems
• Open-source contributions
• Kafka or streaming technologies
• Feature engineering best practices`;

// ─── Components ────────────────────────────────────────────────────────────────

function ScoreRing({ score }) {
  const r = 52, cx = 64, cy = 64;
  const circ = 2 * Math.PI * r;
  const dash = (score / 100) * circ;
  const color = score >= 75 ? "#16a34a" : score >= 50 ? "#d97706" : "#dc2626";
  const label = score >= 75 ? "High" : score >= 50 ? "Medium" : "Low";
  return (
    <div style={{display:"flex",flexDirection:"column",alignItems:"center",gap:8}}>
      <svg width="128" height="128" viewBox="0 0 128 128">
        <circle cx={cx} cy={cy} r={r} fill="none" stroke="#e5e7eb" strokeWidth="10"/>
        <circle cx={cx} cy={cy} r={r} fill="none" stroke={color} strokeWidth="10"
          strokeDasharray={`${dash} ${circ}`} strokeLinecap="round"
          transform="rotate(-90 64 64)" style={{transition:"stroke-dasharray 1s ease"}}/>
        <text x={cx} y={cy-8} textAnchor="middle" fontSize="26" fontWeight="700" fill={color}>{score}</text>
        <text x={cx} y={cy+14} textAnchor="middle" fontSize="11" fill="#6b7280">/ 100</text>
      </svg>
      <span style={{fontSize:12,color:"#6b7280",letterSpacing:"0.08em",textTransform:"uppercase",fontWeight:600}}>{label} match</span>
    </div>
  );
}

function HireabilityBadge({ level }) {
  const cfg = {
    High: { bg:"#dcfce7", color:"#15803d", border:"#86efac", icon:"▲" },
    Medium: { bg:"#fef9c3", color:"#92400e", border:"#fde68a", icon:"◆" },
    Low: { bg:"#fee2e2", color:"#991b1b", border:"#fca5a5", icon:"▼" }
  }[level] || {};
  return (
    <div style={{display:"inline-flex",alignItems:"center",gap:6,padding:"6px 14px",borderRadius:20,background:cfg.bg,border:`1.5px solid ${cfg.border}`,fontWeight:700,fontSize:14,color:cfg.color,letterSpacing:"0.05em"}}>
      <span style={{fontSize:10}}>{cfg.icon}</span> {level} Hireability
    </div>
  );
}

function SkillPill({ skill, variant }) {
  const styles = {
    matched: { bg:"#dcfce7",color:"#15803d",border:"#86efac" },
    missing: { bg:"#fee2e2",color:"#991b1b",border:"#fca5a5" },
    bonus:   { bg:"#dbeafe",color:"#1e40af",border:"#93c5fd" }
  }[variant];
  return (
    <span style={{display:"inline-block",padding:"3px 10px",borderRadius:12,fontSize:12,fontWeight:500,
      background:styles.bg,color:styles.color,border:`1px solid ${styles.border}`,margin:"3px 4px 3px 0"}}>
      {skill}
    </span>
  );
}

function HighlightedText({ text, keywords }) {
  if (!keywords || keywords.length === 0) return <span>{text}</span>;
  const kwSet = new Set(keywords.map(k => k.toLowerCase()));
  const parts = text.split(/(\b\w+\b)/);
  return (
    <span>
      {parts.map((part, i) =>
        kwSet.has(part.toLowerCase())
          ? <mark key={i} style={{background:"#fef9c3",borderRadius:3,padding:"1px 2px",color:"#92400e"}}>{part}</mark>
          : <span key={i}>{part}</span>
      )}
    </span>
  );
}

function StatCard({ label, value, sub }) {
  return (
    <div style={{background:"#f9fafb",borderRadius:10,padding:"14px 16px",border:"0.5px solid #e5e7eb",flex:1,minWidth:120}}>
      <div style={{fontSize:22,fontWeight:700,color:"#111827",lineHeight:1}}>{value}</div>
      <div style={{fontSize:12,color:"#6b7280",marginTop:4}}>{label}</div>
      {sub && <div style={{fontSize:11,color:"#9ca3af",marginTop:2}}>{sub}</div>}
    </div>
  );
}

function ProgressBar({ label, value, max, color="#4f46e5" }) {
  const pct = Math.round((value / Math.max(max,1)) * 100);
  return (
    <div style={{marginBottom:10}}>
      <div style={{display:"flex",justifyContent:"space-between",fontSize:13,marginBottom:4}}>
        <span style={{color:"#374151",fontWeight:500}}>{label}</span>
        <span style={{color:"#6b7280"}}>{value}/{max}</span>
      </div>
      <div style={{height:7,borderRadius:4,background:"#e5e7eb",overflow:"hidden"}}>
        <div style={{height:"100%",width:`${pct}%`,background:color,borderRadius:4,transition:"width 0.8s ease"}}/>
      </div>
    </div>
  );
}

// ─── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [resume, setResume] = useState("");
  const [jobDesc, setJobDesc] = useState("");
  const [result, setResult] = useState(null);
  const [aiInsights, setAiInsights] = useState(null);
  const [loading, setLoading] = useState(false);
  const [aiLoading, setAiLoading] = useState(false);
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState("overview");
  const [showHighlight, setShowHighlight] = useState(false);

  const loadSample = useCallback(() => {
    setResume(SAMPLE_RESUME);
    setJobDesc(SAMPLE_JD);
    setResult(null);
    setAiInsights(null);
    setError("");
  }, []);

  const analyze = useCallback(async () => {
    if (!resume.trim() || !jobDesc.trim()) { setError("Both fields are required."); return; }
    if (resume.trim().length < 50) { setError("Resume is too short."); return; }
    if (jobDesc.trim().length < 50) { setError("Job description is too short."); return; }
    setError("");
    setLoading(true);
    setResult(null);
    setAiInsights(null);
    setActiveTab("overview");

    await new Promise(r => setTimeout(r, 300));
    const res = matchResume(resume, jobDesc);
    setResult(res);
    setLoading(false);

    setAiLoading(true);
    try {
      const ai = await getAIAnalysis(resume, jobDesc, res);
      setAiInsights(ai);
    } catch (e) { /* AI insights optional */ }
    setAiLoading(false);
  }, [resume, jobDesc]);

  const scoreColor = result ? (result.matchScore >= 75 ? "#16a34a" : result.matchScore >= 50 ? "#d97706" : "#dc2626") : "#6366f1";

  return (
    <div style={{fontFamily:"'IBM Plex Mono','Fira Mono',monospace",maxWidth:900,margin:"0 auto",padding:"24px 16px",color:"#111827"}}>

      {/* Header */}
      <div style={{marginBottom:32,borderBottom:"2px solid #111827",paddingBottom:16}}>
        <div style={{display:"flex",alignItems:"center",gap:12,marginBottom:4}}>
          <div style={{width:36,height:36,background:"#111827",borderRadius:8,display:"flex",alignItems:"center",justifyContent:"center"}}>
            <span style={{color:"#fff",fontSize:18}}>▣</span>
          </div>
          <h1 style={{margin:0,fontSize:22,fontWeight:700,letterSpacing:"-0.02em"}}>Smart Resume Matcher</h1>
          <span style={{marginLeft:"auto",fontSize:11,color:"#6b7280",background:"#f3f4f6",padding:"3px 8px",borderRadius:6,border:"0.5px solid #e5e7eb"}}>AI-powered · NLP</span>
        </div>
        <p style={{margin:0,fontSize:13,color:"#6b7280",fontFamily:"system-ui,sans-serif"}}>TF-IDF cosine similarity + skill taxonomy extraction + Claude AI insights</p>
      </div>

      {/* Input Section */}
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16,marginBottom:16}}>
        <div>
          <label style={{display:"block",fontSize:12,fontWeight:600,color:"#374151",marginBottom:6,letterSpacing:"0.08em",textTransform:"uppercase"}}>Resume Text</label>
          <textarea
            value={resume}
            onChange={e => setResume(e.target.value)}
            placeholder="Paste your resume here..."
            style={{width:"100%",height:220,padding:"12px",fontSize:12,fontFamily:"inherit",borderRadius:10,border:"1.5px solid #d1d5db",resize:"vertical",boxSizing:"border-box",outline:"none",lineHeight:1.6,background:"#fafafa",color:"#111827"}}
          />
          <div style={{fontSize:11,color:"#9ca3af",marginTop:4}}>{resume.split(/\s+/).filter(Boolean).length} words</div>
        </div>
        <div>
          <label style={{display:"block",fontSize:12,fontWeight:600,color:"#374151",marginBottom:6,letterSpacing:"0.08em",textTransform:"uppercase"}}>Job Description</label>
          <textarea
            value={jobDesc}
            onChange={e => setJobDesc(e.target.value)}
            placeholder="Paste the job description here..."
            style={{width:"100%",height:220,padding:"12px",fontSize:12,fontFamily:"inherit",borderRadius:10,border:"1.5px solid #d1d5db",resize:"vertical",boxSizing:"border-box",outline:"none",lineHeight:1.6,background:"#fafafa",color:"#111827"}}
          />
          <div style={{fontSize:11,color:"#9ca3af",marginTop:4}}>{jobDesc.split(/\s+/).filter(Boolean).length} words</div>
        </div>
      </div>

      {error && <div style={{background:"#fee2e2",color:"#991b1b",padding:"10px 14px",borderRadius:8,fontSize:13,marginBottom:12,border:"1px solid #fca5a5"}}>{error}</div>}

      <div style={{display:"flex",gap:10,marginBottom:32}}>
        <button onClick={analyze} disabled={loading}
          style={{padding:"10px 24px",background:loading?"#6b7280":"#111827",color:"#fff",border:"none",borderRadius:8,fontSize:13,fontWeight:600,cursor:loading?"not-allowed":"pointer",fontFamily:"inherit",letterSpacing:"0.03em",transition:"background 0.2s"}}>
          {loading ? "Analyzing..." : "▶ Analyze Match"}
        </button>
        <button onClick={loadSample}
          style={{padding:"10px 18px",background:"transparent",border:"1.5px solid #d1d5db",borderRadius:8,fontSize:13,cursor:"pointer",fontFamily:"inherit",color:"#374151"}}>
          Load Sample
        </button>
      </div>

      {/* Results */}
      {result && (
        <div style={{animation:"fadeIn 0.4s ease"}}>

          {/* Score hero row */}
          <div style={{display:"flex",gap:20,marginBottom:24,flexWrap:"wrap",alignItems:"center",background:"#f9fafb",border:"1.5px solid #e5e7eb",borderRadius:14,padding:"20px 24px"}}>
            <ScoreRing score={result.matchScore} />
            <div style={{flex:1,minWidth:200}}>
              <HireabilityBadge level={result.hireability} />
              <p style={{margin:"12px 0 8px",fontSize:13,color:"#374151",fontFamily:"system-ui,sans-serif",lineHeight:1.6}}>{result.hireReason}</p>
              <p style={{margin:0,fontSize:12,color:"#6b7280",fontFamily:"system-ui,sans-serif",fontStyle:"italic",lineHeight:1.5}}>{result.recommendation}</p>
            </div>
            <div style={{display:"flex",gap:10,flexWrap:"wrap"}}>
              <StatCard label="Matched Skills" value={result.matched.length} sub={`of ${result.stats.jdSkills} required`}/>
              <StatCard label="Missing Skills" value={result.missing.length} sub="to acquire"/>
              <StatCard label="Bonus Skills" value={result.bonus.length} sub="extra you have"/>
            </div>
          </div>

          {/* Tabs */}
          <div style={{display:"flex",gap:4,marginBottom:16,borderBottom:"1.5px solid #e5e7eb",paddingBottom:0}}>
            {["overview","skills","keywords","ai-coach"].map(tab => (
              <button key={tab} onClick={() => setActiveTab(tab)}
                style={{padding:"8px 16px",fontSize:12,fontWeight:600,fontFamily:"inherit",cursor:"pointer",border:"none",background:"transparent",
                  color:activeTab===tab?"#111827":"#6b7280",borderBottom:activeTab===tab?"2.5px solid #111827":"2.5px solid transparent",
                  textTransform:"uppercase",letterSpacing:"0.07em",transition:"all 0.15s",marginBottom:-1.5}}>
                {tab==="ai-coach"?"✦ AI Coach":tab}
              </button>
            ))}
          </div>

          {/* Tab: Overview */}
          {activeTab==="overview" && (
            <div style={{animation:"fadeIn 0.3s ease"}}>
              <div style={{marginBottom:20}}>
                <h3 style={{fontSize:13,fontWeight:600,textTransform:"uppercase",letterSpacing:"0.08em",color:"#374151",marginBottom:12}}>Skill Coverage</h3>
                <ProgressBar label="Matched Skills" value={result.matched.length} max={result.stats.jdSkills} color="#16a34a"/>
                <ProgressBar label="Missing Skills" value={result.missing.length} max={result.stats.jdSkills} color="#dc2626"/>
                <ProgressBar label="Resume Skills Found" value={result.stats.resumeSkills} max={Math.max(result.stats.resumeSkills,20)} color="#4f46e5"/>
              </div>
              <div style={{background:"#fffbeb",border:"1px solid #fde68a",borderRadius:10,padding:"14px 16px"}}>
                <div style={{fontSize:12,fontWeight:600,color:"#92400e",marginBottom:6,textTransform:"uppercase",letterSpacing:"0.07em"}}>Recommendation</div>
                <p style={{margin:0,fontSize:13,color:"#78350f",fontFamily:"system-ui,sans-serif",lineHeight:1.6}}>{result.recommendation}</p>
              </div>
            </div>
          )}

          {/* Tab: Skills */}
          {activeTab==="skills" && (
            <div style={{animation:"fadeIn 0.3s ease"}}>
              <div style={{marginBottom:20}}>
                <h3 style={{fontSize:12,fontWeight:600,textTransform:"uppercase",letterSpacing:"0.08em",color:"#15803d",marginBottom:8}}>✓ Matched Skills ({result.matched.length})</h3>
                {result.matched.length ? result.matched.map(s => <SkillPill key={s} skill={s} variant="matched"/>) : <span style={{fontSize:13,color:"#6b7280"}}>None found</span>}
              </div>
              <div style={{marginBottom:20}}>
                <h3 style={{fontSize:12,fontWeight:600,textTransform:"uppercase",letterSpacing:"0.08em",color:"#991b1b",marginBottom:8}}>✗ Missing Skills ({result.missing.length})</h3>
                {result.missing.length ? result.missing.map(s => <SkillPill key={s} skill={s} variant="missing"/>) : <span style={{fontSize:13,color:"#6b7280"}}>None — great!</span>}
              </div>
              <div>
                <h3 style={{fontSize:12,fontWeight:600,textTransform:"uppercase",letterSpacing:"0.08em",color:"#1e40af",marginBottom:8}}>+ Bonus Skills on Resume ({result.bonus.length})</h3>
                {result.bonus.length ? result.bonus.map(s => <SkillPill key={s} skill={s} variant="bonus"/>) : <span style={{fontSize:13,color:"#6b7280"}}>None extra</span>}
              </div>
            </div>
          )}

          {/* Tab: Keywords */}
          {activeTab==="keywords" && (
            <div style={{animation:"fadeIn 0.3s ease"}}>
              <div style={{display:"flex",alignItems:"center",gap:12,marginBottom:16}}>
                <h3 style={{fontSize:12,fontWeight:600,textTransform:"uppercase",letterSpacing:"0.08em",color:"#374151",margin:0}}>Shared Keywords ({result.sharedKeywords.length})</h3>
                <button onClick={() => setShowHighlight(!showHighlight)}
                  style={{fontSize:11,padding:"4px 12px",borderRadius:6,border:"1.5px solid #d1d5db",background:showHighlight?"#111827":"transparent",color:showHighlight?"#fff":"#374151",cursor:"pointer",fontFamily:"inherit",fontWeight:600}}>
                  {showHighlight ? "Hide Highlights" : "Highlight in Text"}
                </button>
              </div>

              <div style={{display:"flex",flexWrap:"wrap",gap:4,marginBottom:20}}>
                {result.sharedKeywords.map(k => (
                  <span key={k} style={{background:"#f0fdf4",color:"#15803d",border:"1px solid #86efac",borderRadius:6,padding:"3px 10px",fontSize:12,fontWeight:500}}>{k}</span>
                ))}
              </div>

              {showHighlight && (
                <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
                  <div>
                    <div style={{fontSize:11,fontWeight:600,textTransform:"uppercase",letterSpacing:"0.08em",color:"#6b7280",marginBottom:8}}>Resume</div>
                    <div style={{fontSize:12,lineHeight:1.8,background:"#fafafa",border:"1px solid #e5e7eb",borderRadius:8,padding:14,maxHeight:280,overflow:"auto",whiteSpace:"pre-wrap",fontFamily:"system-ui,sans-serif"}}>
                      <HighlightedText text={resume} keywords={result.sharedKeywords}/>
                    </div>
                  </div>
                  <div>
                    <div style={{fontSize:11,fontWeight:600,textTransform:"uppercase",letterSpacing:"0.08em",color:"#6b7280",marginBottom:8}}>Job Description</div>
                    <div style={{fontSize:12,lineHeight:1.8,background:"#fafafa",border:"1px solid #e5e7eb",borderRadius:8,padding:14,maxHeight:280,overflow:"auto",whiteSpace:"pre-wrap",fontFamily:"system-ui,sans-serif"}}>
                      <HighlightedText text={jobDesc} keywords={result.sharedKeywords}/>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Tab: AI Coach */}
          {activeTab==="ai-coach" && (
            <div style={{animation:"fadeIn 0.3s ease"}}>
              {aiLoading && (
                <div style={{display:"flex",alignItems:"center",gap:10,padding:"20px",color:"#6b7280",fontSize:13}}>
                  <div style={{width:18,height:18,border:"2px solid #e5e7eb",borderTop:"2px solid #6366f1",borderRadius:"50%",animation:"spin 0.8s linear infinite"}}/>
                  Claude is analyzing your profile...
                </div>
              )}
              {!aiLoading && aiInsights && (
                <div style={{display:"flex",flexDirection:"column",gap:14}}>
                  {aiInsights.executive_summary && (
                    <div style={{background:"#f0f9ff",border:"1px solid #bae6fd",borderRadius:10,padding:"14px 16px"}}>
                      <div style={{fontSize:11,fontWeight:700,textTransform:"uppercase",letterSpacing:"0.08em",color:"#0369a1",marginBottom:6}}>Executive Summary</div>
                      <p style={{margin:0,fontSize:13,color:"#0c4a6e",fontFamily:"system-ui,sans-serif",lineHeight:1.6}}>{aiInsights.executive_summary}</p>
                    </div>
                  )}
                  <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
                    {aiInsights.strengths?.length > 0 && (
                      <div style={{background:"#f0fdf4",border:"1px solid #bbf7d0",borderRadius:10,padding:"14px 16px"}}>
                        <div style={{fontSize:11,fontWeight:700,textTransform:"uppercase",letterSpacing:"0.08em",color:"#15803d",marginBottom:8}}>Your Strengths</div>
                        {aiInsights.strengths.map((s,i) => <div key={i} style={{fontSize:13,color:"#166534",fontFamily:"system-ui,sans-serif",marginBottom:5,display:"flex",gap:6}}><span>✓</span>{s}</div>)}
                      </div>
                    )}
                    {aiInsights.gaps?.length > 0 && (
                      <div style={{background:"#fff7ed",border:"1px solid #fed7aa",borderRadius:10,padding:"14px 16px"}}>
                        <div style={{fontSize:11,fontWeight:700,textTransform:"uppercase",letterSpacing:"0.08em",color:"#c2410c",marginBottom:8}}>Gaps to Address</div>
                        {aiInsights.gaps.map((g,i) => <div key={i} style={{fontSize:13,color:"#9a3412",fontFamily:"system-ui,sans-serif",marginBottom:5,display:"flex",gap:6}}><span>△</span>{g}</div>)}
                      </div>
                    )}
                  </div>
                  {aiInsights.action_items?.length > 0 && (
                    <div style={{background:"#faf5ff",border:"1px solid #e9d5ff",borderRadius:10,padding:"14px 16px"}}>
                      <div style={{fontSize:11,fontWeight:700,textTransform:"uppercase",letterSpacing:"0.08em",color:"#7c3aed",marginBottom:8}}>Action Items</div>
                      {aiInsights.action_items.map((a,i) => (
                        <div key={i} style={{fontSize:13,color:"#5b21b6",fontFamily:"system-ui,sans-serif",marginBottom:6,display:"flex",gap:8}}>
                          <span style={{fontWeight:700,minWidth:18}}>{i+1}.</span>{a}
                        </div>
                      ))}
                    </div>
                  )}
                  {aiInsights.resume_tips?.length > 0 && (
                    <div style={{background:"#fefce8",border:"1px solid #fde68a",borderRadius:10,padding:"14px 16px"}}>
                      <div style={{fontSize:11,fontWeight:700,textTransform:"uppercase",letterSpacing:"0.08em",color:"#92400e",marginBottom:8}}>Resume Tips</div>
                      {aiInsights.resume_tips.map((t,i) => <div key={i} style={{fontSize:13,color:"#78350f",fontFamily:"system-ui,sans-serif",marginBottom:5,display:"flex",gap:6}}><span>→</span>{t}</div>)}
                    </div>
                  )}
                  {aiInsights.interview_prep && (
                    <div style={{background:"#f0fdf4",border:"1px solid #86efac",borderRadius:10,padding:"12px 16px",display:"flex",gap:10,alignItems:"flex-start"}}>
                      <span style={{fontSize:16}}>🎯</span>
                      <div>
                        <div style={{fontSize:11,fontWeight:700,textTransform:"uppercase",letterSpacing:"0.08em",color:"#15803d",marginBottom:4}}>Interview Prep Focus</div>
                        <div style={{fontSize:13,color:"#166534",fontFamily:"system-ui,sans-serif"}}>{aiInsights.interview_prep}</div>
                      </div>
                    </div>
                  )}
                </div>
              )}
              {!aiLoading && !aiInsights && (
                <div style={{color:"#6b7280",fontSize:13,fontFamily:"system-ui,sans-serif",padding:20,textAlign:"center"}}>
                  AI coaching not available — check your API connection.
                </div>
              )}
            </div>
          )}
        </div>
      )}

      <style>{`
        @keyframes fadeIn { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)} }
        @keyframes spin { to{transform:rotate(360deg)} }
        textarea:focus { border-color: #6366f1 !important; box-shadow: 0 0 0 3px rgba(99,102,241,0.1); }
      `}</style>
    </div>
  );
}
