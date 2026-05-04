from flask import Flask, request, jsonify
from flask_cors import CORS
from matcher import match_resume
import os

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def index():
    return """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Smart Resume Matcher</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Courier New',monospace;background:#0f172a;color:#e2e8f0;min-height:100vh;padding:32px 16px}
.wrap{max-width:920px;margin:0 auto}
h1{font-size:22px;color:#818cf8;margin-bottom:4px;letter-spacing:-0.02em}
.sub{color:#64748b;font-size:12px;margin-bottom:28px}
label{display:block;font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:#64748b;margin-bottom:6px;font-weight:700}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:14px}
textarea{width:100%;height:190px;background:#1e293b;color:#e2e8f0;border:1.5px solid #334155;border-radius:8px;padding:12px;font-family:inherit;font-size:12px;resize:vertical;outline:none;line-height:1.6}
textarea:focus{border-color:#818cf8}
.wc{font-size:10px;color:#475569;margin-top:4px}
.btns{display:flex;gap:10px;margin-bottom:28px}
.btn-primary{padding:10px 24px;background:#6366f1;color:#fff;border:none;border-radius:8px;font-family:inherit;font-size:12px;font-weight:700;cursor:pointer;letter-spacing:.03em;transition:background .2s}
.btn-primary:hover{background:#4f46e5}
.btn-primary:disabled{background:#334155;cursor:not-allowed}
.btn-sec{padding:10px 16px;background:transparent;border:1.5px solid #334155;border-radius:8px;font-family:inherit;font-size:12px;color:#94a3b8;cursor:pointer}
.btn-sec:hover{border-color:#64748b}
#error{background:#450a0a;color:#fca5a5;border:1px solid #7f1d1d;padding:10px 14px;border-radius:8px;font-size:12px;margin-bottom:14px;display:none}
#results{display:none}
.hero{background:#1e293b;border:1.5px solid #334155;border-radius:12px;padding:20px 24px;margin-bottom:14px;display:flex;align-items:center;gap:24px;flex-wrap:wrap}
.ring-wrap{display:flex;flex-direction:column;align-items:center;gap:6px}
.ring-label{font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:#64748b;font-weight:700}
.hero-info{flex:1;min-width:180px}
.badge{display:inline-flex;align-items:center;gap:5px;padding:5px 14px;border-radius:20px;font-size:12px;font-weight:700;margin-bottom:10px}
.High{background:#14532d;color:#86efac;border:1px solid #166534}
.Medium{background:#451a03;color:#fde68a;border:1px solid #92400e}
.Low{background:#450a0a;color:#fca5a5;border:1px solid #7f1d1d}
.rec{font-size:12px;color:#94a3b8;line-height:1.6;font-style:italic}
.stats{display:flex;gap:10px;flex-wrap:wrap}
.stat{background:#0f172a;border-radius:8px;padding:10px 14px;text-align:center;min-width:80px;border:1px solid #1e293b}
.stat-num{font-size:20px;font-weight:700;color:#818cf8}
.stat-lbl{font-size:10px;color:#64748b;margin-top:2px;text-transform:uppercase;letter-spacing:.06em}
.tabs{display:flex;gap:0;border-bottom:1.5px solid #334155;margin-bottom:16px}
.tab{padding:8px 16px;background:transparent;border:none;border-bottom:2.5px solid transparent;font-family:inherit;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#64748b;cursor:pointer;transition:all .15s;margin-bottom:-1.5px}
.tab.on{color:#818cf8;border-bottom-color:#818cf8}
.panel{display:none}.panel.on{display:block}
.card{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:14px 16px;margin-bottom:12px}
.sec{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px}
.sec.g{color:#86efac}.sec.r{color:#fca5a5}.sec.b{color:#93c5fd}.sec.y{color:#fde68a}
.pill{display:inline-block;padding:3px 10px;border-radius:10px;font-size:11px;font-weight:600;margin:2px 3px 2px 0}
.pm{background:#14532d;color:#86efac;border:1px solid #166534}
.px{background:#450a0a;color:#fca5a5;border:1px solid #7f1d1d}
.pb{background:#1e3a8a;color:#93c5fd;border:1px solid #1d4ed8}
.pk{background:#1e3a5f;color:#7dd3fc;border:1px solid #0284c7}
.prog-wrap{margin-bottom:10px}
.prog-row{display:flex;justify-content:space-between;font-size:11px;margin-bottom:3px;color:#94a3b8}
.prog-bg{height:6px;border-radius:4px;background:#334155;overflow:hidden}
.prog-fill{height:100%;border-radius:4px;transition:width 1s ease}
mark{background:#854d0e;color:#fef9c3;border-radius:3px;padding:1px 2px}
.hl-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}
.hl-box{background:#0f172a;border:1px solid #334155;border-radius:8px;padding:12px;font-size:11px;line-height:1.8;max-height:260px;overflow:auto;white-space:pre-wrap;font-family:inherit;color:#cbd5e1}
</style>
</head>
<body>
<div class="wrap">
  <h1>▣ Smart Resume Matcher</h1>
  <div class="sub">TF-IDF &middot; Cosine Similarity &middot; Skill Taxonomy &middot; NLP Engine</div>

  <div id="error"></div>

  <div class="grid">
    <div>
      <label>Resume Text</label>
      <textarea id="resume" placeholder="Paste your full resume here..." oninput="wc('resume','rwc')"></textarea>
      <div class="wc" id="rwc">0 words</div>
    </div>
    <div>
      <label>Job Description</label>
      <textarea id="jd" placeholder="Paste the job description here..." oninput="wc('jd','jwc')"></textarea>
      <div class="wc" id="jwc">0 words</div>
    </div>
  </div>

  <div class="btns">
    <button class="btn-primary" id="abtn" onclick="analyze()">&#9654; Analyze Match</button>
    <button class="btn-sec" onclick="loadSample()">Load Sample Data</button>
  </div>

  <div id="results">
    <!-- Hero -->
    <div class="hero">
      <div class="ring-wrap">
        <svg width="110" height="110" viewBox="0 0 128 128">
          <circle cx="64" cy="64" r="52" fill="none" stroke="#1e293b" stroke-width="10"/>
          <circle id="arc" cx="64" cy="64" r="52" fill="none" stroke="#818cf8" stroke-width="10"
            stroke-dasharray="0 326.73" stroke-linecap="round" transform="rotate(-90 64 64)"
            style="transition:stroke-dasharray 1s ease"/>
          <text id="snum" x="64" y="56" text-anchor="middle" font-size="26" font-weight="700" fill="#818cf8" font-family="Courier New">0</text>
          <text x="64" y="74" text-anchor="middle" font-size="11" fill="#64748b" font-family="Courier New">/ 100</text>
        </svg>
        <div class="ring-label" id="match-lbl">— match</div>
      </div>
      <div class="hero-info">
        <div class="badge High" id="hbadge">— Hireability</div>
        <div class="rec" id="hreason" style="margin-bottom:8px"></div>
        <div class="rec" id="rec"></div>
      </div>
      <div class="stats">
        <div class="stat"><div class="stat-num" id="sm">0</div><div class="stat-lbl">Matched</div></div>
        <div class="stat"><div class="stat-num" style="color:#fca5a5" id="sx">0</div><div class="stat-lbl">Missing</div></div>
        <div class="stat"><div class="stat-num" style="color:#93c5fd" id="sb">0</div><div class="stat-lbl">Bonus</div></div>
      </div>
    </div>

    <!-- Tabs -->
    <div class="tabs">
      <button class="tab on" onclick="tab('overview',this)">Overview</button>
      <button class="tab" onclick="tab('skills',this)">Skills</button>
      <button class="tab" onclick="tab('keywords',this)">Keywords</button>
    </div>

    <!-- Overview -->
    <div class="panel on" id="p-overview">
      <div class="card">
        <div class="sec y">Skill Coverage</div>
        <div id="prog-m" class="prog-wrap"></div>
        <div id="prog-x" class="prog-wrap"></div>
        <div id="prog-r" class="prog-wrap"></div>
      </div>
      <div class="card" style="background:#1a1200;border-color:#854d0e">
        <div class="sec y">Recommendation</div>
        <div id="rec2" style="font-size:12px;color:#fde68a;line-height:1.6"></div>
      </div>
    </div>

    <!-- Skills -->
    <div class="panel" id="p-skills">
      <div class="card">
        <div class="sec g" id="lm">✓ Matched Skills</div>
        <div id="pills-m"></div>
      </div>
      <div class="card">
        <div class="sec r" id="lx">✗ Missing Skills</div>
        <div id="pills-x"></div>
      </div>
      <div class="card">
        <div class="sec b" id="lb">+ Bonus Skills on Resume</div>
        <div id="pills-b"></div>
      </div>
    </div>

    <!-- Keywords -->
    <div class="panel" id="p-keywords">
      <div class="card">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
          <div class="sec b" id="lkw" style="margin:0">Shared Keywords</div>
          <button class="btn-sec" onclick="toggleHL()" id="hlbtn" style="font-size:10px;padding:3px 10px">Highlight in Text</button>
        </div>
        <div id="pills-kw"></div>
        <div class="hl-grid" id="hl-grid" style="display:none">
          <div>
            <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px">Resume</div>
            <div class="hl-box" id="hl-resume"></div>
          </div>
          <div>
            <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px">Job Description</div>
            <div class="hl-box" id="hl-jd"></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
var DATA={};
var hlOn=false;

function wc(tid,wid){
  var t=document.getElementById(tid).value.split(/\s+/).filter(Boolean).length;
  document.getElementById(wid).textContent=t+' words';
}

function loadSample(){
  document.getElementById('resume').value=`SARAH CHEN - Senior Software Engineer
SUMMARY
Software engineer 6 years experience building web apps and data pipelines. Machine learning and cloud-native architectures. Strong leadership in cross-functional agile teams.

SKILLS
Python, JavaScript, TypeScript, SQL, React, FastAPI, Django, TensorFlow, scikit-learn, pandas, numpy, AWS, Docker, Kubernetes, GitHub Actions, Terraform, PostgreSQL, MongoDB, Redis, BigQuery, machine learning, deep learning, NLP, feature engineering, model deployment, MLOps, data pipeline, ETL, airflow, dbt, A/B testing, agile, scrum, mentoring, collaboration, leadership, communication

EXPERIENCE
Senior Software Engineer | DataFlow Inc | 2022-Present
Real-time data pipeline 500K events/day using Python Kafka AWS.
ML models churn prediction 87% accuracy scikit-learn TensorFlow.
React FastAPI platform 50K monthly active users.
Mentored 4 engineers agile sprint ceremonies cut delivery 30%.
CI/CD GitHub Actions Docker Kubernetes deployments.

Software Engineer | Nexus Tech | 2019-2021
Python Django React full stack features ecommerce platform.
PostgreSQL database schemas 2 million records.
ETL pipelines Apache Airflow dbt business intelligence dashboards.
A/B testing framework statistics Python collaboration.

EDUCATION: BS Computer Science UC Berkeley 2019
PROJECTS: NLP toolkit 500 GitHub stars huggingface transformers langchain`;

  document.getElementById('jd').value=`Senior ML Engineer - AI Platform Team TechCorp AI

REQUIRED
5 plus years software engineering 2 plus years ML engineering.
Strong Python programming TypeScript JavaScript plus.
ML frameworks TensorFlow PyTorch scikit-learn XGBoost.
Cloud platforms AWS GCP Azure.
Docker Kubernetes containerization orchestration.
SQL data warehouses BigQuery Redshift Snowflake.
Data pipelines Apache Airflow Spark dbt.
NLP large language models LLM MLOps practices.
Leadership communication agile scrum teamwork.

NICE TO HAVE
LangChain HuggingFace RAG systems fine-tuning.
Open source contributions.
Kafka streaming feature engineering statistics.
Model deployment deep learning computer vision.`;

  wc('resume','rwc'); wc('jd','jwc');
  document.getElementById('results').style.display='none';
}

async function analyze(){
  var r=document.getElementById('resume').value.trim();
  var j=document.getElementById('jd').value.trim();
  var err=document.getElementById('error');
  if(!r||!j){err.textContent='Both fields are required.';err.style.display='block';return;}
  if(r.length<50){err.textContent='Resume text is too short.';err.style.display='block';return;}
  err.style.display='none';
  var btn=document.getElementById('abtn');
  btn.disabled=true;btn.textContent='Analyzing...';

  try{
    var res=await fetch('/api/match',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({resume:r,job_description:j})
    });
    var d=await res.json();
    if(d.error){err.textContent=d.error;err.style.display='block';btn.disabled=false;btn.textContent='\\u25B6 Analyze Match';return;}
    DATA=d;DATA.rTxt=r;DATA.jTxt=j;
    render(d);
  }catch(e){
    err.textContent='Could not reach the API. Make sure Flask is running on port 5000.';
    err.style.display='block';
  }
  btn.disabled=false;btn.textContent='\\u25B6 Analyze Match';
}

function render(d){
  var circ=2*Math.PI*52;
  var dash=(d.match_score/100)*circ;
  var col=d.match_score>=75?'#86efac':d.match_score>=50?'#fde68a':'#fca5a5';
  document.getElementById('arc').style.strokeDasharray=dash+' '+circ;
  document.getElementById('arc').setAttribute('stroke',col);
  document.getElementById('snum').textContent=d.match_score;
  document.getElementById('snum').setAttribute('fill',col);
  document.getElementById('match-lbl').textContent=(d.match_score>=75?'High':d.match_score>=50?'Medium':'Low')+' match';

  var hb=document.getElementById('hbadge');
  hb.textContent=d.hireability+' Hireability';
  hb.className='badge '+d.hireability;
  document.getElementById('hreason').textContent=d.hireability_reason;
  document.getElementById('rec').textContent=d.recommendation;
  document.getElementById('rec2').textContent=d.recommendation;
  document.getElementById('sm').textContent=d.matched_skills.length;
  document.getElementById('sx').textContent=d.missing_skills.length;
  document.getElementById('sb').textContent=d.bonus_skills.length;

  var jSk=d.matched_skills.length+d.missing_skills.length||1;
  prog('prog-m','Matched Skills',d.matched_skills.length,jSk,'#86efac');
  prog('prog-x','Missing Skills',d.missing_skills.length,jSk,'#fca5a5');
  prog('prog-r','Your Resume Skills',d.stats.resume_skills_count,Math.max(d.stats.resume_skills_count,20),'#818cf8');

  document.getElementById('lm').textContent='✓ Matched Skills ('+d.matched_skills.length+')';
  document.getElementById('lx').textContent='✗ Missing Skills ('+d.missing_skills.length+')';
  document.getElementById('lb').textContent='+ Bonus Skills ('+d.bonus_skills.length+')';
  pills('pills-m',d.matched_skills,'pm');
  pills('pills-x',d.missing_skills,'px');
  pills('pills-b',d.bonus_skills,'pb');

  document.getElementById('lkw').textContent='Shared Keywords ('+d.shared_keywords.length+')';
  pills('pills-kw',d.shared_keywords,'pk');

  hlOn=false;
  document.getElementById('hl-grid').style.display='none';
  document.getElementById('hlbtn').textContent='Highlight in Text';

  document.getElementById('results').style.display='block';
  tab('overview',document.querySelector('.tab'));
}

function prog(id,label,val,max,col){
  var pct=Math.round((val/Math.max(max,1))*100);
  document.getElementById(id).innerHTML='<div class="prog-row"><span>'+label+'</span><span>'+val+'/'+max+'</span></div><div class="prog-bg"><div class="prog-fill" style="width:'+pct+'%;background:'+col+'"></div></div>';
}

function pills(id,arr,cls){
  document.getElementById(id).innerHTML=arr&&arr.length?arr.map(function(s){return'<span class="pill '+cls+'">'+s+'</span>';}).join(''):'<span style="font-size:12px;color:#475569">None found</span>';
}

function tab(name,el){
  document.querySelectorAll('.tab').forEach(function(t){t.classList.remove('on');});
  if(el)el.classList.add('on');
  document.querySelectorAll('.panel').forEach(function(p){p.classList.remove('on');p.style.display='none';});
  var p=document.getElementById('p-'+name);
  if(p){p.classList.add('on');p.style.display='block';}
}

function toggleHL(){
  hlOn=!hlOn;
  document.getElementById('hlbtn').textContent=hlOn?'Hide Highlights':'Highlight in Text';
  var g=document.getElementById('hl-grid');
  if(hlOn&&DATA.shared_keywords){
    var kws=DATA.shared_keywords.map(function(k){return k.toLowerCase();});
    function hl(txt){
      return txt.split(/(\\b\\w+\\b)/g).map(function(p){
        return kws.indexOf(p.toLowerCase())>=0?'<mark>'+p+'</mark>':p.replace(/</g,'&lt;');
      }).join('');
    }
    document.getElementById('hl-resume').innerHTML=hl(DATA.rTxt||'');
    document.getElementById('hl-jd').innerHTML=hl(DATA.jTxt||'');
    g.style.display='grid';
  } else {
    g.style.display='none';
  }
}
</script>
</body>
</html>"""


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "Smart Resume Matcher AI"})


@app.route("/api/match", methods=["POST"])
def match():
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400
    resume_text = data.get("resume", "").strip()
    job_text = data.get("job_description", "").strip()
    if not resume_text:
        return jsonify({"error": "resume field is required"}), 400
    if not job_text:
        return jsonify({"error": "job_description field is required"}), 400
    if len(resume_text) < 50:
        return jsonify({"error": "Resume text too short (min 50 chars)"}), 400
    if len(job_text) < 50:
        return jsonify({"error": "Job description too short (min 50 chars)"}), 400
    try:
        result = match_resume(resume_text, job_text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Smart Resume Matcher API on port {port}")
    print(f"Open your browser at: http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
