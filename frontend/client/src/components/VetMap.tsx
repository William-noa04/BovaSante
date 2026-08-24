import { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import icon2x from "leaflet/dist/images/marker-icon-2x.png";
import icon from "leaflet/dist/images/marker-icon.png";
import shadow from "leaflet/dist/images/marker-shadow.png";
import type { Vet } from "@/types/vet";

const vetIcon = L.icon({
  iconUrl: icon,
  iconRetinaUrl: icon2x,
  shadowUrl: shadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
});
const userIcon = L.divIcon({ className: "vet-map-user-marker", html: "<span></span>", iconSize: [18, 18] });

function escapeHtml(value: string) {
  return value.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c] as string);
}

interface VetMapProps {
  center: { lat: number; lon: number };
  vets: Vet[];
  selectedId: string | null;
  fallbackName: string;
}

export function VetMap({ center, vets, selectedId, fallbackName }: VetMapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<L.Map | null>(null);
  const markersRef = useRef<Map<string, L.Marker>>(new Map());

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = L.map(containerRef.current, { scrollWheelZoom: false });
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
      maxZoom: 19,
    }).addTo(map);
    mapRef.current = map;
    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    markersRef.current.forEach((marker) => marker.remove());
    markersRef.current.clear();

    L.marker([center.lat, center.lon], { icon: userIcon }).addTo(map);

    vets.forEach((vet) => {
      const popup = `<strong>${escapeHtml(vet.name || fallbackName)}</strong>${vet.address ? `<br>${escapeHtml(vet.address)}` : ""}`;
      const marker = L.marker([vet.lat, vet.lon], { icon: vetIcon }).addTo(map).bindPopup(popup);
      markersRef.current.set(vet.id, marker);
    });

    const points: [number, number][] = [[center.lat, center.lon], ...vets.map((v): [number, number] => [v.lat, v.lon])];
    if (points.length > 1) map.fitBounds(points, { padding: [30, 30], maxZoom: 14 });
    else map.setView([center.lat, center.lon], 13);

    setTimeout(() => map.invalidateSize(), 0);
  }, [center, vets, fallbackName]);

  useEffect(() => {
    if (!selectedId) return;
    const marker = markersRef.current.get(selectedId);
    if (marker) {
      marker.openPopup();
      mapRef.current?.panTo(marker.getLatLng());
    }
  }, [selectedId]);

  return <div ref={containerRef} className="vet-map" />;
}
