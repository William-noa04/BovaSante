export class ApiError extends Error {
  constructor(message: string, public readonly status?: number) { super(message); }
}

export async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = "Le service est momentanément indisponible.";
    try {
      const body: unknown = await response.json();
      if (typeof body === "object" && body !== null && "detail" in body) {
        const raw = (body as { detail: unknown }).detail;
        detail = typeof raw === "string" ? raw : "Les données envoyées ne sont pas valides.";
      }
    } catch { /* message fallback */ }
    throw new ApiError(detail, response.status);
  }
  return response.json() as Promise<T>;
}
