# Foresight — ADC Conjugation Literature Intelligence 
# Date: 2025-10-14
# Features: PubMed + Crossref, MRI disambiguation, method-only filters,
# TF-IDF semantic ranking, near-duplicate removal, NumPy K-Means clustering,
# extractive summaries, signals, and action brief — all with lightweight deps.

import re, string, math, requests, numpy as np, pandas as pd, streamlit as st
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import time, random

# --- PubMed request helper with retry and API key support ---
def eutils_request(endpoint, params, max_retries=5):
    """Handles NCBI E-Utilities calls with retry/backoff and API key."""
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    headers = {"User-Agent": "Syngene-Foresight/1.0 (contact: research@syngeneintl.com)"}

    # Add your API key automatically from Streamlit Secrets
    try:
        api_key = st.secrets.get("NCBI_API_KEY", None)
    except Exception:
        api_key = None
    if api_key:
        params["api_key"] = api_key

    for attempt in range(max_retries):
        r = requests.get(f"{base}/{endpoint}", params=params, headers=headers, timeout=30)
        if r.status_code == 429:  # too many requests
            time.sleep(min(10, (2 ** attempt)) + random.uniform(0, 0.5))  # polite wait
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()
    return r

# --------------------------- Config ---------------------------
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
CROSSREF = "https://api.crossref.org/works"
HEADERS = {"User-Agent": "Syngene-Foresight/1.0 (contact: research@syngeneintl.com)"}

IMAGING_BAD = ["magnetic resonance","mri","diffusion","apparent diffusion coefficient","dwi","adc map","mr imaging","diffusion-weighted"]
BIOCONJ_MUST = [
    "antibody-drug conjugate","antibody drug conjugate","antibody conjugate","bioconjugate",
    "conjugation","linker","payload","dar","site-specific","site specific","hydrophilic",
    "cleavable","click","tetrazine","tco","dbco","maleimide","smcc","glycan","sortase",
    "transglutaminase","rebridging","oxime","aldehyde","hic","tff","diafiltration",
    "hydrophobic interaction chromatography"
]
SCALE_UP_CUES = [
    "process development","scale-up","manufacturing","gmp","cmc","qbd","pat","validation",
    "cpp","ipc","dar control","tff","hic","single-use","hpapi","containment","hold time"
]
TARGETS  = ["HER2","TROP2","EGFR","CD19","CD22","Nectin-4","BCMA","PSMA","FOLR1","CLDN18.2","MUC1","GPRC5D","CD79b"]
LINKERS = [
    "val-cit","valine-citrulline","vc-pabc","pabc","para-aminobenzyl carbamate","self-immolative",
    "cathepsin-cleavable","cathepsin","glucuronide","beta-glucuronidase","β-glucuronidase",
    "hydrazone","disulfide","noncleavable","non-cleavable","thioether",
    "pegylated linker","hydrophilic linker","maleimide","smcc","sulfo-smcc","mcc",
    "click chemistry","spaac","strain-promoted","dbco","azide-alkyne","tetrazine","tco",
    "oxime","oxime ligation"
]
PAYLOADS = ["MMAE","MMAF","DM1","DM4","SN-38","PBD","duocarmycin","maytansine","auristatin","camptothecin"]
METHODS  = ["site-specific conjugation","site specific","site-specific","enzymatic conjugation","transglutaminase",
            "sortase","sortase a","glycan engineering","glycoengineering","thiomab","engineered cysteine",
            "cysteine rebridging","aldehyde tag","formylglycine","oxime ligation","bioorthogonal","thioether"]
REAGENTS = [
    "smcc","sulfo-smcc","mcc","dbco","tco","tetrazine","azide","nhs ester","nhs-ester","maleimide",
    "thiol","sulfhydryl","n3","spaac","iodoacetamide","nem","n-ethylmaleimide","tcep","dtt",
    "traut's reagent","2-iminothiolane","spdp","lc-smcc","sata","hydroxylamine","imidazole",
    "phosphate buffer","carbonate buffer","borate buffer","dmsa","dmso","dma","dmf"
]
CLINICAL_TERMS = ["phase I","phase II","phase III","randomized","open-label","double-blind"]
AI_TERMS = ["machine learning","deep learning","transformer","artificial intelligence","AI"]

# ------------------------ Text utils -------------------------
def clean_text(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.lower()
    s = re.sub(r"[\n\r\t]", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip()

def any_kw(text, kws):
    t = text.lower()
    return any(k in t for k in kws)

def word_boundary_count(text, vocab):
    txt = (text or "").lower()
    counts = {}
    for term in vocab:
        t = term.lower()
        pattern = (
            r'(?<![a-z0-9])' +
            re.escape(t)
              .replace(r'\-', r'[- ]?')   # match hyphen OR space
              .replace('beta', r'(beta|β)') +  # beta or β
            r's?(?![a-z0-9])'             # optional plural
        )
        c = len(re.findall(pattern, txt))
        if c > 0:
            counts[term] = c
    return counts


def is_bioconj_paper(title, abstract):
    txt = f"{title} {abstract}".lower()
    if any(b in txt for b in IMAGING_BAD):
        if not any(s in txt for s in ["antibody-drug conjugate","antibody drug conjugate","bioconjugate"]):
            return False
    return any(k in txt for k in BIOCONJ_MUST)

# --------------------- PubMed / Crossref ---------------------

def pm_esearch(term, start_date, end_date, retmax=200):
    """Search PubMed IDs for given term and time window (with retry & API key)."""
    p = {
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": str(retmax),
        "sort": "pubdate",
        "datetype": "pdat",
        "mindate": start_date.strftime("%Y/%m/%d"),
        "maxdate": end_date.strftime("%Y/%m/%d"),
    }
    r = eutils_request("esearch.fcgi", p)
    return r.json().get("esearchresult", {}).get("idlist", [])


def pm_esummary(ids, chunk=180):
    """Fetch PubMed summaries in safe chunks to avoid 414 or 429 errors."""
    if not ids:
        return []
    out = []
    for i in range(0, len(ids), chunk):
        batch = ids[i:i + chunk]
        r = eutils_request("esummary.fcgi", {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "json"
        })
        data = r.json().get("result", {})
        for pmid in batch:
            rec = data.get(pmid, {})
            if not rec:
                continue
            doi = ""
            for aid in rec.get("articleids", []):
                if aid.get("idtype") == "doi":
                    doi = aid.get("value")
                    break
            out.append({
                "PMID": pmid,
                "Title": rec.get("title", ""),
                "Journal": rec.get("source", ""),
                "PubDate": rec.get("pubdate", ""),
                "Authors": ", ".join([a.get("name", "") for a in rec.get("authors", [])][:5]),
                "DOI": doi,
                "PubMedLink": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "DOILink": f"https://doi.org/{doi}" if doi else ""
            })
        time.sleep(0.3)  # polite pause
    return out


def pm_efetch_abs(ids):
    """Fetch PubMed abstracts with retry/backoff."""
    absd = {}
    for pmid in ids:
        try:
            r = eutils_request("efetch.fcgi", {
                "db": "pubmed",
                "id": pmid,
                "retmode": "text",
                "rettype": "abstract"
            })
            absd[pmid] = r.text.strip()
        except Exception:
            absd[pmid] = ""
        time.sleep(0.2)  # small delay to avoid 429
    return absd


def crossref_cites(doi):
    """Get citation counts from Crossref."""
    if not doi:
        return None
    try:
        r = requests.get(f"{CROSSREF}/{doi}", headers={"User-Agent": "Syngene-Foresight-Agent/1.0"}, timeout=20)
        if r.status_code != 200:
            return None
        return r.json().get("message", {}).get("is-referenced-by-count")
    except Exception:
        return None

# ------------------ TF-IDF + semantic ranking ------------------
def build_vocab(texts, max_terms=6000, min_len=3):
    counts={}
    for t in texts:
        for w in clean_text(t).split():
            if len(w) < min_len: continue
            counts[w] = counts.get(w,0) + 1
    vocab = [w for w,_ in sorted(counts.items(), key=lambda x:x[1], reverse=True)[:max_terms]]
    index = {w:i for i,w in enumerate(vocab)}
    return vocab, index

def tfidf_matrix(texts, index):
    mat = np.zeros((len(texts), len(index)), dtype=np.float32)
    for i,t in enumerate(texts):
        for w in clean_text(t).split():
            j = index.get(w)
            if j is not None:
                mat[i,j] += 1.0
    df = (mat > 0).sum(axis=0)
    idf = np.log((1 + len(texts)) / (1 + df)) + 1.0
    mat = mat * idf
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
    return mat / norms

def embed_texts(texts):
    vocab, index = build_vocab(texts)
    return tfidf_matrix(texts, index), (vocab, index)

def embed_query(q, meta):
    vocab, index = meta
    return tfidf_matrix([q], index)[0]

def cosine_topk(qv, M, k=None):
    sims = M @ qv
    order = np.argsort(-sims)
    if k is None: k = len(order)
    return order[:k], sims[order[:k]]

# --------------- Near-duplicate removal (titles) ---------------
def dedupe_titles(titles, threshold=0.9):
    sets=[set(clean_text(t).split()) for t in titles]
    keep=[]; removed=set()
    for i in range(len(titles)):
        if i in removed: continue
        keep.append(i)
        Ai = sets[i]
        for j in range(i+1,len(titles)):
            if j in removed: continue
            Aj = sets[j]
            inter = len(Ai & Aj); union = len(Ai | Aj) or 1
            if inter / union >= threshold:
                removed.add(j)
    return keep

# ------------------------ NumPy K-Means ------------------------
def kmeans_numpy(X, k, iters=25, seed=42):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.choice(n, size=k, replace=False)
    C = X[idx].copy()
    for _ in range(iters):
        d = 1.0 - (X @ C.T)  # cosine distance (X,C normalized)
        labels = np.argmin(d, axis=1)
        newC = np.zeros_like(C)
        for i in range(k):
            pts = X[labels==i]
            if len(pts)==0:
                newC[i] = X[rng.integers(0,n)]
            else:
                v = pts.mean(axis=0)
                nv = np.linalg.norm(v) + 1e-8
                newC[i] = v / nv
        if np.allclose(newC, C, atol=1e-4):
            break
        C = newC
    return labels, C

def choose_k(n):
    return max(2, min(6, int(math.sqrt(max(2, n)/2) + 1)))

def label_clusters(texts, labels, topn=5):
    vocab, index = build_vocab(texts, max_terms=4000)
    M = tfidf_matrix(texts, index)
    names=[]
    for c in range(labels.max()+1):
        idx = np.where(labels==c)[0]
        if len(idx)==0:
            names.append(f"Cluster {c+1}"); continue
        mean = M[idx].mean(axis=0)
        top_idx = np.argsort(-mean)[:topn]
        inv = {i:w for w,i in index.items()}
        names.append(", ".join([inv[i] for i in top_idx]))
    return names

# -------------------- Summaries / Signals ---------------------
def best_sentences(text, n=3):
    if not text: return []
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sents) <= n: return sents
    cues = METHODS + LINKERS + PAYLOADS + REAGENTS + SCALE_UP_CUES + ["dar","hic","tff","diafiltration","aggregation","stability","efficacy"]
    scores=[]
    for s in sents:
        sl = s.lower(); base=0
        for t in cues:
            patt = r'\b' + re.escape(t.lower()).replace(r'\-', r'[- ]?') + r'\b'
            base += len(re.findall(patt, sl))
        wlen = len(s.split()); length_bonus = 1.0 if 8 <= wlen <= 40 else 0.6
        scores.append(base * length_bonus)
    order = np.argsort(-np.array(scores))[:n]
    return [sents[i].strip() for i in order]

def paper_summary(title, abstract):
    bullets = best_sentences(abstract or title, n=3)
    if not bullets: return "—"
    novelty = ["novel","first","optimized","hydrophilic linker","vc-pabc","val-cit","tetrazine","tco","dbco","smcc","site-specific","rebridging"]
    tag = " (new/notable)" if any(w in (abstract or "").lower() for w in novelty) else ""
    return " • ".join(bullets) + tag

def build_signals(df):
    # Weight abstracts 2× so linker/reagent terms get more visibility
    titles = df["Title"].fillna("")
    abstracts = df["Abstract"].fillna("")
    full = "\n".join((titles + " " + abstracts + " " + abstracts).tolist())

    def topn(d, k=5):
        return sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]

    return {
        "methods":  topn(word_boundary_count(full, METHODS)),
        "linkers":  topn(word_boundary_count(full, LINKERS)),
        "reagents": topn(word_boundary_count(full, REAGENTS)),
        "payloads": topn(word_boundary_count(full, PAYLOADS)),
        "targets":  topn(word_boundary_count(full, TARGETS)),
        "clinical": topn(word_boundary_count(full, CLINICAL_TERMS)),
        "ai":       topn(word_boundary_count(full, AI_TERMS)),
    }

def brief(domain, start_date, end_date, df, sig):
    window = f"{start_date.strftime('%d %b %Y')} - {end_date.strftime('%d %b %Y')}"
    what = [
        "Methods: "  + (", ".join([f"{k} ({v})" for k,v in sig["methods"]]) if sig["methods"] else "-"),
        "Linkers: "  + (", ".join([f"{k} ({v})" for k,v in sig["linkers"]]) if sig["linkers"] else "-"),
        "Reagents: " + (", ".join([f"{k} ({v})" for k,v in sig["reagents"]]) if sig["reagents"] else "-"),
        "Payloads: " + (", ".join([f"{k} ({v})" for k,v in sig["payloads"]]) if sig["payloads"] else "-"),
        "Targets: "  + (", ".join([f"{k} ({v})" for k,v in sig["targets"]]) if sig["targets"] else "-"),
    ]
    why = "Supports GMP/CMC/QbD readiness: DAR control, HIC/TFF, solvent/linker choices, and HPAPI handling."
    nexts = [
        "Define CPPs/IPC around DAR, residuals, aggregation.",
        "Benchmark HIC→TFF→polish purification for yield/stability.",
        "Document solvent/linker handling and single-use/HPAPI containment.",
    ]
    return (f"**Insight Brief — {domain}**  \n**Window:** {window}  \n**Papers analyzed:** {len(df)}\n\n"
            f"**What is happening**\n- " + "\n- ".join(what) + "\n\n"
            f"**Why it matters**\n{why}\n\n"
            f"**What to do next**\n- " + "\n- ".join(nexts))

# -------------------------- Streamlit UI ----------------------
st.set_page_config(page_title="Foresight – ADC Literature Intelligence", layout="wide")
st.title("Foresight — ADC Literature Intelligence")
st.caption("Method focus · MRI disambiguation · TF-IDF semantic ranking · NumPy K-Means clustering · Source links")

with st.sidebar:
    st.header("Scan settings")
    default_query = (
    '("antibody-drug conjugate"[TIAB] OR "antibody drug conjugate"[TIAB] OR "Antibody-Drug Conjugates"[MeSH]) '
    'AND (conjugation OR "site-specific" OR "site specific" OR linker OR "val-cit" OR "vc-PABC" OR PABC '
    'OR glucuronide OR "β-glucuronidase" OR hydrazone OR disulfide OR noncleavable OR "non-cleavable" '
    'OR maleimide OR SMCC OR "sulfo-SMCC" OR MCC OR "click chemistry" OR SPAAC OR DBCO OR azide OR tetrazine OR TCO '
    'OR "NHS-ester" OR "NHS ester" OR TCEP OR DTT OR NEM OR iodoacetamide) '
    'NOT (MRI OR "magnetic resonance" OR diffusion OR "apparent diffusion coefficient" OR DWI OR "ADC map")'
)
    domain = st.text_input("PubMed query", value=default_query)
    months_back = st.number_input("Time window (months)", 1, 36, 24, 1)
    retmax = st.slider("Max PubMed items to fetch", 50, 800, 400, 50)
    topn_citations = st.slider("Top N by citations (pre-filter)", 10, 150, 60, 5)
    final_k = st.slider("Final N after ranking & filters", 10, 80, 30, 5)
    debug = st.checkbox("Debug mode (show counts/logs)", value=True)
    run = st.button("Scan")

if run:
    end_date = datetime.utcnow()
    start_date = end_date - relativedelta(months=int(months_back))

    with st.spinner("Searching PubMed…"):
        try:
            import re
            from datetime import datetime, timedelta

            # --- Get sidebar values safely ---
            months_raw = st.session_state.get("months", 12)
            domain = st.session_state.get("domain", "ADC conjugation methods")
            retmax_raw = st.session_state.get("retmax", 200)
            debug = st.session_state.get("debug", False)

            # Convert sidebar inputs to proper type
            months = int(months_raw) if str(months_raw).isdigit() else 12
            retmax = int(retmax_raw) if str(retmax_raw).isdigit() else 200

            # Define how far back to search
            today = datetime.utcnow()
            start_date = today - timedelta(days=30 * months)
            end_date = today

            if debug:
                st.info(f"Searching PubMed for '{domain}' between {start_date} and {end_date}")

            # --- Perform PubMed search ---
            ids = pm_esearch(domain, start_date, end_date, retmax=retmax)
            if debug:
                st.info(f"PubMed IDs found: {len(ids)}")

            if not ids:
                st.warning("No PubMed IDs found for this query/time window.")
                st.stop()

            # --- Get summaries ---
            meta = pm_esummary(ids)
            if debug:
                st.info(f"Summaries fetched: {len(meta)}")

            if not meta:
                st.warning("No summaries returned by PubMed for these IDs.")
                st.stop()

            # --- Fetch abstracts ---
            pmids = [m["PMID"] for m in meta]
            abstracts = pm_efetch_abs(pmids)

        except Exception as e:
            st.error(f"PubMed error: {e}")
            st.stop()

        # --- Get summaries (safe chunked fetch) ---
        meta = pm_esummary(ids)
        if debug: st.info(f"Summaries fetched: {len(meta)}")

        if not meta:
            st.warning("No summaries returned by PubMed for these IDs.")
            st.stop()

        # --- Limit Crossref lookups to most recent N ---
        RECENT_FOR_CITES = 250
        def _parse_year(pd_str):
            m = re.search(r"(\d{4})", pd_str or "")
            return int(m.group(1)) if m else 0

        meta_sorted = sorted(meta, key=lambda m: _parse_year(m.get("PubDate", "")), reverse=True)
        recent_pmids_for_cites = set([m["PMID"] for m in meta_sorted[:RECENT_FOR_CITES]])

        # --- Fetch abstracts for all PMIDs ---
        pmids = [m["PMID"] for m in meta]
        abstracts = pm_efetch_abs(pmids)

    except Exception as e:
        st.error(f"PubMed error: {e}")
        st.stop()

rows = []
with st.spinner("Fetching citations & filtering…"):
    for m in meta:
        title = m.get("Title","")
        abs_  = abstracts.get(m["PMID"], "")
        if not is_bioconj_paper(title, abs_):
            continue
        if not any_kw(title + " " + abs_, BIOCONJ_MUST):
            continue

        want_cite = m["PMID"] in recent_pmids_for_cites
        has_doi   = bool(m.get("DOI",""))
        cites = crossref_cites(m["DOI"]) if (want_cite and has_doi) else None

        rows.append({
            **m,
            "Abstract": abs_,
            "Citations": cites if (cites is not None) else -1
        })

    if debug: st.info(f"After method/MRI filters: {len(rows)}")
    df = pd.DataFrame(rows)
    if df.empty:
        st.warning("No method-focused papers found. Adjust query or widen timeline.")
        st.stop()

    # Pre-rank by citations
    df = df.sort_values(by=["Citations","PubDate"], ascending=[False,False]).head(topn_citations)
    if debug: st.info(f"After citation pre-rank: {len(df)}")

    # TF-IDF semantic ranking towards scale-up seed
    seed = ("ADC conjugation method development; site-specific; linkers; payload; DAR control; HIC; TFF; GMP; "
            "CMC; QbD; PAT; scale-up manufacturing; solvent handling; HPAPI containment")
    texts = (df["Title"].fillna("") + ". " + df["Abstract"].fillna("")).tolist()
    M, meta_embed = embed_texts(texts)
    qv = embed_query(seed, meta_embed)
    order, sims = cosine_topk(qv, M, k=len(df))
    df_sem = df.iloc[order].copy()
    df_sem["SemanticScore"] = sims

    # Near-duplicate removal
    keep = dedupe_titles(df_sem["Title"].fillna("").tolist(), threshold=0.88)
    df_sem = df_sem.iloc[keep].head(final_k).reset_index(drop=True)
    if debug: st.info(f"Final selected (semantic + dedupe): {len(df_sem)}")

    # Clustering (NumPy K-Means)
    st.subheader("Selected papers (semantically ranked)")
    st.dataframe(df_sem[["Title","Journal","PubDate","Authors","Citations","SemanticScore","PMID","DOI","PubMedLink","DOILink"]],
                 use_container_width=True)

    st.subheader("Themes (clusters)")
    if len(df_sem) >= 2:
        X = M[order][keep][:len(df_sem)]  # vectors for selected items
        k = choose_k(len(df_sem))
        labels, _ = kmeans_numpy(X, k=k, iters=30, seed=42)
        df_sem["ThemeID"] = labels
        theme_names = label_clusters((df_sem["Title"] + ". " + df_sem["Abstract"]).fillna("").tolist(),
                                     labels, topn=5)
        df_sem["Theme"] = [theme_names[i] for i in labels]
        st.write("Detected themes:", ", ".join(sorted(set(df_sem["Theme"]))))
    else:
        df_sem["ThemeID"] = 0
        df_sem["Theme"] = "All"
        st.write("Detected themes: All")

    # Signals & brief
    st.subheader("Signals snapshot")
    sig = build_signals(df_sem)
    def dictdf(name,pairs): return pd.DataFrame(pairs, columns=[name,"count"]) if pairs else pd.DataFrame(columns=[name,"count"])
    c1,c2,c3 = st.columns(3)
    with c1: st.markdown("**Methods**");  st.dataframe(dictdf("method", sig["methods"]), use_container_width=True, height=220)
    with c2: st.markdown("**Linkers**");  st.dataframe(dictdf("linker", sig["linkers"]), use_container_width=True, height=220)
    with c3: st.markdown("**Reagents**"); st.dataframe(dictdf("reagent", sig["reagents"]), use_container_width=True, height=220)
    c4,c5,c6 = st.columns(3)
    with c4: st.markdown("**Payloads**"); st.dataframe(dictdf("payload", sig["payloads"]), use_container_width=True, height=220)
    with c5: st.markdown("**Targets**");  st.dataframe(dictdf("target", sig["targets"]), use_container_width=True, height=220)
    with c6: st.markdown("**Clinical**"); st.dataframe(dictdf("clinical", sig["clinical"]), use_container_width=True, height=220)

    st.subheader("Insight Brief")
    st.markdown(brief("ADC conjugation method development & scale-up", start_date, end_date, df_sem, sig))

    # Paper summaries
    st.subheader("Paper summaries (extractive)")
    sums=[]
    for _,r in df_sem.iterrows():
        sums.append({
            "Theme": r.get("Theme",""),
            "Title": r["Title"],
            "Summary (problem · method · result / novelty)": paper_summary(r["Title"], r.get("Abstract","")),
            "Citations": r.get("Citations",""),
            "Journal": r.get("Journal",""),
            "PubDate": r.get("PubDate",""),
            "PubMedLink": r.get("PubMedLink",""),
            "DOILink": r.get("DOILink","")
        })
    df_sum = pd.DataFrame(sums)
    st.dataframe(df_sum[["Theme","Title","Summary (problem · method · result / novelty)","Citations","Journal","PubDate","PubMedLink","DOILink"]],
                 use_container_width=True, height=500)

    # Downloads
    st.subheader("Download")
    st.download_button("Download Selected (CSV)", df_sem.to_csv(index=False).encode("utf-8"),
                       file_name="selected_papers.csv", mime="text/csv")
    st.download_button("Download Summaries (CSV)", df_sum.to_csv(index=False).encode("utf-8"),
                       file_name="paper_summaries.csv", mime="text/csv")

st.markdown("---")
st.caption("No heavy ML deps. TF-IDF + NumPy K-Means provide semantic ranking and themes suitable for method scouting.")
