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

function digitsOnly(phone: string): string {
  const trimmed = phone.trim().replace(/^00/, "+");
  return trimmed.replace(/[^\d+]/g, "");
}

export function telHref(phone: string): string {
  return `tel:${digitsOnly(phone)}`;
}

export function whatsappHref(phone: string): string {
  return `https://wa.me/${digitsOnly(phone).replace(/\D/g, "")}`;
}