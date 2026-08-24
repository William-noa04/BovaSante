import { FormEvent, useEffect, useState } from "react";
import { LoaderCircle, MapPin, MessageCircle, Navigation, Phone, Search } from "lucide-react";
import { useLanguage } from "@/contexts/LanguageContext";
import { geocodePlace, searchNearby, telHref, whatsappHref } from "@/services/vetApi";
import { VetMap } from "@/components/VetMap";
import type { TranslationKey } from "@/lib/i18n";
import type { Vet } from "@/types/vet";
import { Empty } from "./Pages";

export function Veterinarians() {
  const { t } = useLanguage();
  const [query, setQuery] = useState("");
  const [vets, setVets] = useState<Vet[] | null>(null);
  const [center, setCenter] = useState<{ lat: number; lon: number } | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [autoLocating, setAutoLocating] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<TranslationKey | null>(null);

  async function fetchNear(lat: number, lon: number) {
    setLoading(true);
    setError(null);
    setSelectedId(null);
    try {
      const results = await searchNearby(lat, lon);
      setVets(results);
      setCenter({ lat, lon });
    } catch {
      setError("vets.error.generic");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (!("geolocation" in navigator)) {
      setAutoLocating(false);
      setError("vets.locationDenied");
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => { setAutoLocating(false); fetchNear(pos.coords.latitude, pos.coords.longitude); },
      () => { setAutoLocating(false); setError("vets.locationDenied"); },
      { timeout: 10000 },
    );
  }, []);

  async function submitSearch(event: FormEvent) {
    event.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const place = await geocodePlace(query.trim());
      await fetchNear(place.lat, place.lon);
    } catch {
      setError("vets.error.geocode");
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <div className="page-title">
        <p className="eyebrow">{t("vets.eyebrow")}</p>
        <h1>{t("vets.title")}</h1>
        <p>{t("vets.subtitle")}</p>
      </div>
      <section className="panel">
        <form className="vet-search" onSubmit={submitSearch}>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={t("vets.search.placeholder")}
          />
          <button className="button primary" type="submit" disabled={loading}><Search />{t("vets.search.submit")}</button>
        </form>

        {autoLocating && <p className="notice"><LoaderCircle className="spin" />{t("vets.locating")}</p>}
        {!autoLocating && loading && <p className="notice"><LoaderCircle className="spin" />{t("vets.loading")}</p>}
        {!loading && error && <p className="notice warning">{t(error)}</p>}

        {!loading && center && (
          <div className="vet-map-wrap">
            <VetMap center={center} vets={vets ?? []} selectedId={selectedId} fallbackName={t("vets.fallbackName")} />
          </div>
        )}

        {!loading && !error && vets !== null && vets.length === 0 && (
          <Empty title={t("vets.empty.title")} text={t("vets.empty.text")} />
        )}

        {!loading && vets !== null && vets.length > 0 && (
          <div className="vet-list">
            {vets.map((vet) => (
              <VetCard key={vet.id} vet={vet} t={t} selected={vet.id === selectedId} onSelect={() => setSelectedId(vet.id)} />
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

function VetCard({ vet, t, selected, onSelect }: {
  vet: Vet; t: (key: TranslationKey) => string; selected: boolean; onSelect: () => void;
}) {
  const name = vet.name || t("vets.fallbackName");
  const distanceLabel = t("vets.distance").replace("{km}", vet.distanceKm.toFixed(1));
  const mapHref = `https://www.openstreetmap.org/?mlat=${vet.lat}&mlon=${vet.lon}#map=16/${vet.lat}/${vet.lon}`;

  return (
    <div className={`vet-card ${selected ? "selected" : ""}`}>
      <button type="button" className="vet-card-main" onClick={onSelect}>
        <strong>{name}</strong>
        {vet.address && <small>{vet.address}</small>}
        <small className="vet-distance"><MapPin />{distanceLabel}</small>
      </button>
      <div className="vet-actions">
        {vet.phone ? (
          <>
            <a className="button primary" href={telHref(vet.phone)}><Phone />{t("vets.call")}</a>
            <a className="button subtle" href={whatsappHref(vet.phone)} target="_blank" rel="noreferrer">
              <MessageCircle />{t("vets.whatsapp")}
            </a>
          </>
        ) : (
          <a className="button subtle" href={mapHref} target="_blank" rel="noreferrer"><Navigation />{t("vets.directions")}</a>
        )}
      </div>
    </div>
  );
}
