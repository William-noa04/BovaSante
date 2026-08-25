import { ChangeEvent, FormEvent, useEffect, useMemo, useState } from "react";
import { Link, useLocation, useRoute, useSearch } from "wouter";
import { Activity, AlertTriangle, Bot, Camera, ChevronRight, ClipboardList, CloudOff, History as HistoryIcon, LoaderCircle, MessageCircle, Plus, Send, ShieldCheck, Sparkles, Stethoscope, Trash2, Upload, X } from "lucide-react";
import { predictionApi } from "@/services/predictionApi";
import { chatbotApi } from "@/services/chatbotApi";
import { ApiError } from "@/services/http";
import { getAnalyses, saveAnalysis } from "@/lib/analysisStore";
import { deleteConversation, getConversations, saveConversation, type StoredConversation } from "@/lib/chatStore";
import { useLanguage } from "@/contexts/LanguageContext";
import type { TranslationKey } from "@/lib/i18n";
import type { ChatHistoryItem } from "@/types/chatbot";
import type { MultimodalPrediction, StoredAnalysis, TabularInput, SimplifiedTabularInput } from "@/types/prediction";

type T = (key: TranslationKey) => string;

const categoricalFields = [
  ["Breed", "Race", "Breed"], ["Region", "Région", "Region"], ["Country", "Pays", "Country"], ["Climate_Zone", "Zone climatique", "Climate zone"],
  ["Management_System", "Système d’élevage", "Farming system"], ["Lactation_Stage", "Stade de lactation", "Lactation stage"], ["Feed_Type", "Type d’alimentation", "Feed type"], ["Season", "Saison", "Season"],
] as const;
const numericFields = [
  ["Age_Months", "Âge (mois)", "Age (months)"], ["Weight_kg", "Poids (kg)", "Weight (kg)"], ["Parity", "Parité", "Parity"], ["Days_in_Milk", "Jours en lactation", "Days in milk"],
  ["Feed_Quantity_kg", "Alimentation (kg)", "Feed quantity (kg)"], ["Water_Intake_L", "Eau bue (L)", "Water intake (L)"], ["Walking_Distance_km", "Distance de marche (km)", "Walking distance (km)"],
  ["Grazing_Duration_hrs", "Pâturage (heures)", "Grazing duration (hrs)"], ["Rumination_Time_hrs", "Rumination (heures)", "Rumination time (hrs)"], ["Resting_Hours", "Repos (heures)", "Resting hours"],
  ["Body_Temperature_C", "Température corporelle (°C)", "Body temperature (°C)"], ["Heart_Rate_bpm", "Fréquence cardiaque (bpm)", "Heart rate (bpm)"], ["Respiratory_Rate", "Fréquence respiratoire", "Respiratory rate"],
  ["Ambient_Temperature_C", "Température ambiante (°C)", "Ambient temperature (°C)"], ["Humidity_percent", "Humidité (%)", "Humidity (%)"], ["Housing_Score", "Score du logement", "Housing score"],
  ["Milk_Yield_L", "Production de lait (L)", "Milk yield (L)"], ["Previous_Week_Avg_Yield", "Production moyenne précédente (L)", "Previous week avg yield (L)"], ["Body_Condition_Score", "État corporel", "Body condition score"],
  ["Milking_Interval_hrs", "Intervalle de traite (h)", "Milking interval (h)"],
] as const;
const VACCINE_FIELDS = [
  ["FMD_Vaccine", "Fièvre aphteuse", "Foot-and-mouth disease"], ["Brucellosis_Vaccine", "Brucellose", "Brucellosis"],
  ["HS_Vaccine", "Septicémie hémorragique", "Hemorrhagic septicemia"], ["BQ_Vaccine", "Charbon symptomatique", "Blackleg"],
  ["Anthrax_Vaccine", "Charbon bactéridien", "Anthrax"], ["IBR_Vaccine", "IBR", "IBR"],
  ["BVD_Vaccine", "BVD", "BVD"], ["Rabies_Vaccine", "Rage", "Rabies"],
] as const;
type FieldName = keyof TabularInput;
type FormValues = Record<string, string>;
const initialForm = (): FormValues => Object.fromEntries([...categoricalFields, ...numericFields].map(([key]) => [key, ""]).concat(VACCINE_FIELDS.map(([key]) => [key, ""])));

function friendlyError(error: unknown, t: T, fallback: TranslationKey) {
  if (error instanceof ApiError) return error.status === 503 ? t("error.modelNotLoaded") : error.message;
  return error instanceof TypeError ? t("error.offline") : t(fallback);
}
function labelDisease(value: string, t: T) {
  return ({ healthy: t("disease.healthy"), lumpy_skin_disease: t("disease.lumpy_skin_disease"), foot_and_mouth_disease: t("disease.foot_and_mouth_disease") } as Record<string, string>)[value] ?? value;
}
function risk(result: MultimodalPrediction, t: T) { if (result.predicted_class === "healthy") return t("risk.low"); return result.confidence >= .75 ? t("risk.high") : t("risk.watch"); }
function pct(value: number) { return `${Math.round(value * 100)} %`; }

export function Dashboard() {
  const { t, lang } = useLanguage();
  const [analyses, setAnalyses] = useState<StoredAnalysis[]>([]);
  const [services, setServices] = useState<"loading" | "ready" | "down">("loading");
  useEffect(() => { setAnalyses(getAnalyses()); predictionApi.health().then(() => setServices("ready")).catch(() => setServices("down")); }, []);
  const detected = analyses.filter(({ result }) => result.predicted_class !== "healthy").length;
  return <div className="page"><section className="hero"><div className="hero-content"><p className="eyebrow light">{t("dashboard.eyebrow")}</p><h1>{t("dashboard.title1")}<br />{t("dashboard.title2")}</h1><p>{t("dashboard.subtitle")}</p>
<div className="hero-actions"><Link className="button cream" href="/analyse"><Camera />{t("dashboard.cta.analyse")}</Link><Link className="button outline" href="/assistant"><MessageCircle />{t("dashboard.cta.assistant")}</Link></div>
  </div><div className="hero-visual" aria-hidden="true"><img src="/hero-cow.jpg" alt=""/></div>
  </section><section className="metric-grid"><Metric icon={<Activity />} label={t("metric.analyses")} value={String(analyses.length)} detail={t("metric.analyses.detail")}/><Metric icon={<AlertTriangle />} label={t("metric.detected")} value={String(detected)} detail={analyses.length ? t("metric.detected.detail") : t("metric.detected.empty")}/><Metric icon={services === "ready" ? <ShieldCheck /> : <CloudOff />} label={t("metric.service")} value={services === "loading" ? "…" : services === "ready" ? t("metric.service.ready") : t("metric.service.down")} detail={t("metric.service.detail")}/></section><DistributionChart analyses={analyses}/><section className="panel"><div className="section-heading"><div><p className="eyebrow">{t("history.eyebrow.local")}</p><h2>{t("history.recent")}</h2></div><Link href="/historique">{t("history.viewAll")} <ChevronRight /></Link></div>{analyses.length ? <div className="analysis-list">{analyses.slice(0, 4).map((item) => <AnalysisRow key={item.id} analysis={item}/>)}</div> : <Empty title={t("history.empty.title")} text={t("history.empty.text")} action={t("history.empty.action")} href="/analyse"/>}</section></div>;
}
const DISEASE_ORDER = ["healthy", "lumpy_skin_disease", "foot_and_mouth_disease"] as const;
const DISEASE_COLORS: Record<string, string> = { healthy: "#2f9469", lumpy_skin_disease: "#c07a2e", foot_and_mouth_disease: "#a83d34" };
function DistributionChart({ analyses }: {analyses: StoredAnalysis[]}) {
  const { t } = useLanguage();
  if (!analyses.length) return <section className="panel"><div className="section-heading"><div><p className="eyebrow">{t("overview.eyebrow")}</p><h2>{t("overview.title")}</h2></div></div><Empty title={t("overview.empty.title")} text={t("overview.empty.text")} action={t("action.newAnalysis")} href="/analyse"/></section>;
  const counts = DISEASE_ORDER.map((cls) => ({ cls, label: labelDisease(cls, t), count: analyses.filter((a) => a.result.predicted_class === cls).length }));
  const max = Math.max(1, ...counts.map((c) => c.count));
  return <section className="panel"><div className="section-heading"><div><p className="eyebrow">{t("overview.eyebrow")}</p><h2>{t("overview.title")}</h2></div></div><div className="distribution-chart">{counts.map(({ cls, label, count }) => <div key={cls} className="distribution-row"><span className="distribution-label"><i style={{ background: DISEASE_COLORS[cls] }}/>{label}</span><i className="distribution-track"><b style={{ width: `${(count / max) * 100}%`, background: DISEASE_COLORS[cls] }}/></i><strong>{count}</strong></div>)}</div></section>;
}
function Metric({ icon, label, value, detail }: {icon: React.ReactNode; label: string; value: string; detail: string}) { return <article className="metric"><span className="icon-badge">{icon}</span><p>{label}</p><strong>{value}</strong><small>{detail}</small></article>; }
function AnalysisRow({ analysis }: { analysis: StoredAnalysis }) { const { t, lang } = useLanguage(); return <Link href={`/bovins/${encodeURIComponent(analysis.cattleId)}`} className="analysis-row"><span className={`dot ${analysis.result.predicted_class === "healthy" ? "safe" : "warning"}`}/><div><strong>{analysis.cattleId}</strong><small>{new Date(analysis.createdAt).toLocaleString(lang === "fr" ? "fr-FR" : "en-US")}</small></div><div className="row-result"><strong>{labelDisease(analysis.result.predicted_class, t)}</strong><small>{pct(analysis.result.confidence)} · {risk(analysis.result, t)}</small></div><ChevronRight /></Link>; }

export function Analysis() {
  const { t } = useLanguage();
  const [location, setLocation] = useLocation();
  const tab: "detailed" | "simplified" = location === "/diagnostic-rapide" ? "simplified" : "detailed";
  return (
    <div className="page">
      <div className="page-title">
        <p className="eyebrow">{t("analysis.eyebrow")}</p>
        <h1>{t("analysisHub.title")}</h1>
        <p>{t("analysisHub.subtitle")}</p>
      </div>
      <div className="tab-switch" role="tablist">
        <button type="button" role="tab" aria-selected={tab === "detailed"} className={tab === "detailed" ? "active" : ""} onClick={() => setLocation("/analyse")}>
          <Stethoscope />{t("analysisHub.tab.detailed")}
        </button>
        <button type="button" role="tab" aria-selected={tab === "simplified"} className={tab === "simplified" ? "active" : ""} onClick={() => setLocation("/diagnostic-rapide")}>
          <Sparkles />{t("analysisHub.tab.simplified")}
        </button>
      </div>
      {tab === "detailed" ? <DetailedForm /> : <SimplifiedForm />}
    </div>
  );
}

function DetailedForm() {
  const { t, lang } = useLanguage();
  const [form, setForm] = useState<FormValues>(initialForm);
  const [cattleId, setCattleId] = useState(""); const [symptoms, setSymptoms] = useState("");
  const [image, setImage] = useState<File | null>(null); const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false); const [error, setError] = useState(""); const [result, setResult] = useState<MultimodalPrediction | null>(null);
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);
  function setImageFile(file: File | null) { setError(""); if (!file) return; if (!/^image\/(jpeg|png|webp)$/.test(file.type)) { setError(t("dropzone.error.type")); return; } if (file.size > 10 * 1024 * 1024) { setError(t("dropzone.error.size")); return; } if (preview) URL.revokeObjectURL(preview); setImage(file); setPreview(URL.createObjectURL(file)); }
  function update(key: string, value: string) { setForm((current) => ({ ...current, [key]: value })); }
  function buildPayload(): TabularInput | null { const empty = [...categoricalFields.map(([key]) => key), ...numericFields.map(([key]) => key), ...VACCINE_FIELDS.map(([key]) => key)].find((key) => !form[key]?.trim()); if (empty) { setError(t("analysis.error.fields")); return null; } return Object.fromEntries(Object.entries(form).map(([key, value]) => [key, categoricalFields.some(([field]) => field === key) ? value.trim() : Number(value)])) as unknown as TabularInput; }
  async function submit(event: FormEvent) { event.preventDefault(); setError(""); setResult(null); if (!cattleId.trim()) { setError(t("analysis.error.cattleId")); return; } if (!image) { setError(t("analysis.error.image")); return; } if (!symptoms.trim()) { setError(t("analysis.error.symptoms")); return; } const payload = buildPayload(); if (!payload) return; setLoading(true); try { const received = await predictionApi.analyzeMultimodal(image, symptoms.trim(), payload); setResult(received); saveAnalysis({ id: crypto.randomUUID(), createdAt: new Date().toISOString(), cattleId: cattleId.trim(), result: received }); } catch (cause) { console.error(cause); setError(friendlyError(cause, t, "analysis.error.generic")); } finally { setLoading(false); } }
  return <><p className="hint form-hint">{t("analysis.subtitle")}</p>{error && <p className="notice error" role="alert">{error}</p>}<form onSubmit={submit} className="analysis-form"><section className="panel"><h2>{t("analysis.step1")}</h2><div className="form-grid"><Field label={t("analysis.cattleId")} value={cattleId} onChange={setCattleId} placeholder={t("analysis.cattleId.placeholder")}/><label className="field wide"><span>{t("analysis.symptoms")}</span><textarea value={symptoms} onChange={(e) => setSymptoms(e.target.value)} placeholder={t("analysis.symptoms.placeholder")} required/></label></div></section><section className="panel"><h2>{t("analysis.step2")}</h2><ImageDropzone preview={preview} onFile={setImageFile} onClear={() => { if (preview) URL.revokeObjectURL(preview); setImage(null); setPreview(null); }}/></section><section className="panel"><h2>{t("analysis.step3")}</h2><p className="hint">{t("analysis.step3.hint")}</p><div className="form-grid">{categoricalFields.map(([key, fr, en]) => <Field key={key} label={lang === "fr" ? fr : en} value={form[key]} onChange={(value) => update(key, value)}/>)}{numericFields.map(([key, fr, en]) => <Field key={key} label={lang === "fr" ? fr : en} type="number" value={form[key]} onChange={(value) => update(key, value)}/>)}</div><div className="vaccine-grid">{VACCINE_FIELDS.map(([key, fr, en]) => <label key={key} className="field"><span>{lang === "fr" ? fr : en}</span><select required value={form[key]} onChange={(e) => update(key, e.target.value)}><option value="">{t("field.choose")}</option><option value="1">{t("field.yes")}</option><option value="0">{t("field.no")}</option></select></label>)}</div></section><button className="button primary submit" disabled={loading}>{loading ? <LoaderCircle className="spin"/> : <Stethoscope/>}{loading ? t("analysis.submitting") : t("analysis.submit")}</button></form>{result && <PredictionResult result={result}/>}</>;
}
function Field({ label, value, onChange, type = "text", placeholder }: {label: string; value: string; onChange: (value: string) => void; type?: string; placeholder?: string}) { return <label className="field"><span>{label}</span><input required type={type} min={type === "number" ? "0" : undefined} value={value} placeholder={placeholder} onChange={(e) => onChange(e.target.value)}/></label>; }
export function ImageDropzone({ preview, onFile, onClear }: {preview: string | null; onFile: (file: File | null) => void; onClear: () => void}) { const { t } = useLanguage(); return <label className={`dropzone ${preview ? "has-image" : ""}`} onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); onFile(e.dataTransfer.files[0] ?? null); }}><input type="file" accept="image/jpeg,image/png,image/webp" onChange={(e: ChangeEvent<HTMLInputElement>) => onFile(e.target.files?.[0] ?? null)}/>{preview ? <><img src={preview} alt={t("dropzone.previewAlt")}/><button type="button" className="remove-image" onClick={(e) => { e.preventDefault(); onClear(); }} aria-label={t("dropzone.remove")}><X /></button></> : <><Upload/><strong>{t("dropzone.cta")}</strong><span>{t("dropzone.hint")}</span></>}</label>; }
const ADVICE_KEY: Record<string, TranslationKey> = {
  healthy: "result.advice.healthy",
  lumpy_skin_disease: "result.advice.lumpy_skin_disease",
  foot_and_mouth_disease: "result.advice.foot_and_mouth_disease",
};
function PredictionResult({ result }: {result: MultimodalPrediction}) {
  const { t } = useLanguage();
  const diseaseLabel = labelDisease(result.predicted_class, t);
  const prefill = t("result.advice.assistantPrefill").replace("{disease}", diseaseLabel).replace("{confidence}", pct(result.confidence));
  return <section className="panel result"><p className="eyebrow">{t("result.eyebrow")}</p><div className="result-main"><div><h2>{t("result.suspicion")}{diseaseLabel}</h2><p>{t("result.confidence")}<strong>{pct(result.confidence)}</strong></p></div><span className={`risk ${result.predicted_class === "healthy" ? "safe" : "warning"}`}>{risk(result, t)}</span></div><h3>{t("result.probabilities")}</h3>{result.probabilities.map(({ label, probability }) => <div className="probability" key={label}><div><span>{labelDisease(label, t)}</span><strong>{pct(probability)}</strong></div><i><b style={{ width: `${Math.round(probability * 100)}%` }}/></i></div>)}{result.warning && <p className="notice warning"><AlertTriangle/>{result.warning}</p>}<div className="advice"><h3>{t("result.advice.title")}</h3><p>{t(ADVICE_KEY[result.predicted_class] ?? "result.advice.healthy")}</p><div className="advice-actions"><Link className="button primary" href="/veterinaires"><Stethoscope/>{t("result.advice.vetCta")}</Link><Link className="button subtle" href={`/assistant?q=${encodeURIComponent(prefill)}`}><MessageCircle/>{t("result.advice.assistantCta")}</Link></div></div><p className="disclaimer">{t("result.disclaimer")}</p></section>;
}

function SimplifiedForm() {
  const { t, lang } = useLanguage();
  const [age, setAge] = useState(""); const [country, setCountry] = useState(""); const [region, setRegion] = useState("");
  const [symptoms, setSymptoms] = useState("");
  const [image, setImage] = useState<File | null>(null); const [preview, setPreview] = useState<string | null>(null);
  const [vaccinesState, setVaccinesState] = useState<Record<string, string>>(Object.fromEntries(VACCINE_FIELDS.map(([key]) => [key, "0"])));
  const [loading, setLoading] = useState(false); const [error, setError] = useState(""); const [result, setResult] = useState<MultimodalPrediction | null>(null);
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);
  function setImageFile(file: File | null) { setError(""); if (!file) return; if (!/^image\/(jpeg|png|webp)$/.test(file.type)) { setError(t("dropzone.error.type")); return; } if (file.size > 10 * 1024 * 1024) { setError(t("dropzone.error.size")); return; } if (preview) URL.revokeObjectURL(preview); setImage(file); setPreview(URL.createObjectURL(file)); }

  async function submit(event: FormEvent) {
    event.preventDefault(); setError(""); setResult(null);
    if (!age.trim() || !country.trim() || !region.trim()) { setError(t("simplified.error.fields")); return; }
    if (!image) { setError(t("simplified.error.image")); return; }
    if (!symptoms.trim()) { setError(t("simplified.error.symptoms")); return; }
    setLoading(true);
    try {
      const payload: SimplifiedTabularInput = {
        Age_Months: Number(age), Country: country.trim(), Region: region.trim(),
        ...Object.fromEntries(VACCINE_FIELDS.map(([key]) => [key, Number(vaccinesState[key])])),
      } as SimplifiedTabularInput;
      const received = await predictionApi.analyzeSimplified(image, symptoms.trim(), payload);
      setResult(received);
      saveAnalysis({ id: crypto.randomUUID(), createdAt: new Date().toISOString(), cattleId: t("simplified.unidentifiedAnimal"), result: received });
    } catch (cause) { console.error(cause); setError(friendlyError(cause, t, "simplified.error.generic")); }
    finally { setLoading(false); }
  }

  return <><p className="hint form-hint">{t("simplified.subtitle")}</p><form onSubmit={submit} className="analysis-form"><section className="panel"><h2>{t("simplified.knownSection")}</h2><div className="form-grid"><label className="field"><span>{t("simplified.age")}</span><input required type="number" min="0" value={age} placeholder={t("simplified.age.placeholder")} onChange={(e) => setAge(e.target.value)}/></label><label className="field"><span>{t("simplified.country")}</span><input required type="text" value={country} placeholder={t("simplified.country.placeholder")} onChange={(e) => setCountry(e.target.value)}/></label><label className="field"><span>{t("simplified.region")}</span><input required type="text" value={region} placeholder={t("simplified.region.placeholder")} onChange={(e) => setRegion(e.target.value)}/></label></div></section><section className="panel"><h2>{t("simplified.vaccinesSection")}</h2><div className="vaccine-grid">{VACCINE_FIELDS.map(([key, fr, en]) => <label key={key} className="field"><span>{lang === "fr" ? fr : en}</span><select value={vaccinesState[key]} onChange={(e) => setVaccinesState((v) => ({ ...v, [key]: e.target.value }))}><option value="0">{t("simplified.vaccine.unknown")}</option><option value="1">{t("field.yes")}</option></select></label>)}</div></section><section className="panel"><h2>{t("simplified.photoSection")}</h2><div className="form-grid"><label className="field wide"><span>{t("simplified.observed")}</span><textarea value={symptoms} onChange={(e) => setSymptoms(e.target.value)} placeholder={t("simplified.observed.placeholder")} required/></label></div><ImageDropzone preview={preview} onFile={setImageFile} onClear={() => { if (preview) URL.revokeObjectURL(preview); setImage(null); setPreview(null); }}/></section><button className="button primary submit" disabled={loading}>{loading ? <LoaderCircle className="spin"/> : <Sparkles/>}{loading ? t("analysis.submitting") : t("simplified.submit")}</button>{error && <p className="notice error" role="alert">{error}</p>}</form>{result && <PredictionResult result={result}/>}</>;
}
export function Assistant() {
  const { t, lang } = useLanguage();
  const search = useSearch();
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [history, setHistory] = useState<ChatHistoryItem[]>([]);
  const [message, setMessage] = useState(() => new URLSearchParams(search).get("q") ?? "");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showHistory, setShowHistory] = useState(false);
  const [conversations, setConversations] = useState<StoredConversation[]>([]);
  const items = useMemo(() => history.filter((item) => item.parts.some((part) => part.text)), [history]);

  function persist(nextHistory: ChatHistoryItem[]) {
    const firstUserText = nextHistory.find((item) => item.role === "user")?.parts.find((part) => part.text)?.text ?? "";
    const now = new Date().toISOString();
    const existing = conversationId ? getConversations().find((c) => c.id === conversationId) : undefined;
    const id = conversationId ?? crypto.randomUUID();
    if (!conversationId) setConversationId(id);
    saveConversation({ id, title: (existing?.title || firstUserText).slice(0, 80), createdAt: existing?.createdAt ?? now, updatedAt: now, history: nextHistory });
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    const text = message.trim();
    if (!text || loading) return;
    setError(""); setLoading(true);
    try {
      const answer = await chatbotApi.send(text, history);
      setHistory(answer.history);
      setMessage("");
      persist(answer.history);
    } catch (cause) { console.error(cause); setError(friendlyError(cause, t, "assistant.error.generic")); }
    finally { setLoading(false); }
  }

  function newChat() { setHistory([]); setConversationId(null); setMessage(""); setError(""); setShowHistory(false); }
  function toggleHistory() { if (!showHistory) setConversations(getConversations()); setShowHistory((v) => !v); }
  function openConversation(conv: StoredConversation) { setHistory(conv.history); setConversationId(conv.id); setError(""); setShowHistory(false); }

  return <div className="page assistant-page"><div className="page-title"><p className="eyebrow">{t("assistant.eyebrow")}</p><h1>{t("assistant.title")}</h1><p>{t("assistant.subtitle")}</p></div><div className="assistant-toolbar"><button type="button" className="button subtle" onClick={newChat}><Plus/>{t("assistant.newChat")}</button><button type="button" className="button subtle" onClick={toggleHistory}><HistoryIcon/>{t("assistant.history")}</button></div>{showHistory && <section className="panel conversation-list">{conversations.length === 0 ? <p className="hint">{t("assistant.history.empty")}</p> : conversations.map((conv) => <div key={conv.id} className={`conversation-row ${conv.id === conversationId ? "active" : ""}`} onClick={() => openConversation(conv)}><div><strong>{conv.title || t("assistant.history.untitled")}</strong><small>{new Date(conv.updatedAt).toLocaleString(lang === "fr" ? "fr-FR" : "en-US")}</small></div><button type="button" className="conversation-delete" onClick={(e) => { e.stopPropagation(); deleteConversation(conv.id); setConversations(getConversations()); if (conv.id === conversationId) newChat(); }} aria-label={t("assistant.history.delete")}><Trash2/></button></div>)}</section>}<section className="chat panel"><div className="messages">{items.length === 0 && <div className="welcome"><Bot/><h2>{t("assistant.welcome.title")}</h2><p>{t("assistant.welcome.text")}</p></div>}{items.map((item, index) => item.parts.filter((part) => part.text).map((part, partIndex) => <p key={`${index}-${partIndex}`} className={`bubble ${item.role === "user" ? "user" : "model"}`}>{part.text}</p>))}{loading && <p className="bubble model"><LoaderCircle className="spin"/> {t("assistant.pending")}</p>}</div>{error && <p className="notice error" role="alert">{error}</p>}<form className="chat-form" onSubmit={submit}><textarea value={message} onChange={(e) => setMessage(e.target.value)} placeholder={t("assistant.placeholder")} aria-label={t("assistant.messageLabel")}/><button className="button primary" disabled={loading || !message.trim()} aria-label={t("assistant.sendLabel")}><Send/>{t("assistant.send")}</button></form></section></div>;
}

export function History() { const { t } = useLanguage(); const [analyses] = useState(getAnalyses); return <div className="page"><div className="page-title"><p className="eyebrow">{t("history.eyebrow")}</p><h1>{t("history.title")}</h1><p>{t("history.subtitle")}</p></div><section className="panel">{analyses.length ? <div className="analysis-list">{analyses.map((analysis) => <AnalysisRow key={analysis.id} analysis={analysis}/>)}</div> : <Empty title={t("history.empty.title2")} text={t("history.empty.text2")} action={t("action.newAnalysis")} href="/analyse"/>}</section></div>; }
export function CattleDetails() {
  const { t, lang } = useLanguage();
  const [, params] = useRoute("/bovins/:id");
  const related = getAnalyses().filter((item) => item.cattleId === params?.id);
  return (
    <div className="page">
      <div className="page-title">
        <p className="eyebrow">{t("cattle.eyebrow")}</p>
        <h1>{params?.id ?? t("cattle.fallback")}</h1>
      </div>
      {related.length ? (
        <div className="cattle-analyses">
          {related.map((analysis) => (
            <div key={analysis.id}>
              <p className="hint cattle-date">{new Date(analysis.createdAt).toLocaleString(lang === "fr" ? "fr-FR" : "en-US")}</p>
              <PredictionResult result={analysis.result} />
            </div>
          ))}
        </div>
      ) : (
        <section className="panel"><Empty title={t("cattle.empty.title")} text={t("cattle.empty.text")} /></section>
      )}
    </div>
  );
}
export function NotFound() { const { t } = useLanguage(); const [, setLocation] = useLocation(); return <div className="page"><section className="panel empty"><h1>{t("notFound.title")}</h1><button className="button primary" onClick={() => setLocation("/")}>{t("notFound.cta")}</button></section></div>; }
export function Empty({ title, text, action, href }: {title: string; text: string; action?: string; href?: string}) { return <div className="empty"><ClipboardList/><h2>{title}</h2><p>{text}</p>{action && href && <Link className="button primary" href={href}>{action}</Link>}</div>; }
