export interface Vet {
  id: string;
  name: string;
  lat: number;
  lon: number;
  address: string | null;
  phone: string | null;
  distanceKm: number;
}

export interface GeocodeResult {
  lat: number;
  lon: number;
  label: string;
}
