import type { GeocodeResult, Vet } from "@/types/vet";

const API_BASE = import.meta.env.VITE_PREDICTION_API_URL;

export function haversineKm(a: { lat: number; lon: number }, b: { lat: number; lon: number }): number {
  const R = 6371;
  const dLat = ((b.lat - a.lat) * Math.PI) / 180;
  const dLon = ((b.lon - a.lon) * Math.PI) / 180;
  const lat1 = (a.lat * Math.PI) / 180;
  const lat2 = (b.lat * Math.PI) / 180;
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(h));
}

function buildAddress(tags: Record<string, string>): string | null {
  const parts = [
    [tags["addr:housenumber"], tags["addr:street"]].filter(Boolean).join(" "),
    tags["addr:city"] || tags["addr:town"] || tags["addr:village"],
  ].filter((part) => part && part.trim());
  return parts.length ? parts.join(", ") : null;
}

interface OverpassElement {
  type: string;
  id: number;
  lat?: number;
  lon?: number;
  center?: { lat: number; lon: number };
  tags?: Record<string, string>;
}

export async function searchNearby(lat: number, lon: number, radiusMeters = 15000): Promise<Vet[]> {
  const response = await fetch(`${API_BASE}/veterinaires/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ lat, lon, radius_meters: radiusMeters }),
  });
  if (!response.ok) throw new Error(`veterinaires/search ${response.status}`);
  const data: { elements: OverpassElement[] } = await response.json();

  const origin = { lat, lon };
  return data.elements
    .map((el): Vet | null => {
      const point = el.type === "node" ? { lat: el.lat, lon: el.lon } : el.center;
      if (!point || point.lat == null || point.lon == null) return null;
      const tags = el.tags ?? {};
      return {
        id: `${el.type}/${el.id}`,
        name: tags.name || "",
        lat: point.lat,
        lon: point.lon,
        address: buildAddress(tags),
        phone: tags.phone || tags["contact:phone"] || null,
        distanceKm: haversineKm(origin, { lat: point.lat, lon: point.lon }),
      };
    })
    .filter((vet): vet is Vet => vet !== null)
    .sort((a, b) => a.distanceKm - b.distanceKm);
}

export async function geocodePlace(query: string): Promise<GeocodeResult> {
  const response = await fetch(`${API_BASE}/veterinaires/geocode?q=${encodeURIComponent(query)}`);
  if (!response.ok) throw new Error(`geocode ${response.status}`);
  const results: { lat: string; lon: string; display_name: string }[] = await response.json();
  if (!results.length) throw new Error("not_found");
  return { lat: parseFloat(results[0].lat), lon: parseFloat(results[0].lon), label: results[0].display_name };
}

/**
 * Numéros OSM bruts, formats très variables : "+33 1 40 26 20 02", "0033146339231",
 * "01 46 33 92 31" (sans indicatif pays), parfois avec la notation "+33 (0)1 ..."
 * où le "(0)" ne fait PAS partie du numéro composable (il indique juste le 0 à
 * garder pour un appel national et à retirer pour un appel international).
 * Renvoie le numéro au format E.164 ("+33146339231") si un indicatif pays est
 * identifiable, sinon null.
 */
function toE164(phone: string): string | null {
  let value = phone.trim().replace(/\(0\)/gi, "");
  value = value.replace(/^00/, "+");
  if (!value.startsWith("+")) return null;
  const digits = value.slice(1).replace(/\D/g, "");
  return digits.length >= 8 ? `+${digits}` : null;
}

export function telHref(phone: string): string {
  const intl = toE164(phone);
  if (intl) return `tel:${intl}`;
  // Pas d'indicatif pays identifiable : on garde le numéro local tel quel,
  // il reste composable depuis un téléphone du même pays (cas très majoritaire).
  return `tel:${phone.trim().replace(/\(0\)/gi, "").replace(/[^\d+]/g, "")}`;
}

/** null si le numéro n'a pas d'indicatif pays identifiable : un lien wa.me sans
 * indicatif pointe vers un mauvais contact (ou aucun) au lieu d'échouer visiblement. */
export function whatsappHref(phone: string): string | null {
  const intl = toE164(phone);
  return intl ? `https://wa.me/${intl.slice(1)}` : null;
}