import { readJson } from "./http";
import type {
  HealthResponse,
  MultimodalPrediction,
  TabularInput,
  SimplifiedTabularInput,
} from "@/types/prediction";

const baseUrl = (import.meta.env.VITE_PREDICTION_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

export const predictionApi = {
  health: () => fetch(`${baseUrl}/health`).then(readJson<HealthResponse>),
  async analyzeMultimodal(image: File, symptoms: string, tabular: TabularInput): Promise<MultimodalPrediction> {
    const data = new FormData();
    data.append("image", image);
    data.append("symptoms", symptoms);
    data.append("tabular_json", JSON.stringify(tabular));
    return readJson<MultimodalPrediction>(await fetch(`${baseUrl}/predict/multimodal`, { method: "POST", body: data }));
  },
  async analyzeSimplified(image: File, symptoms: string, tabular: SimplifiedTabularInput): Promise<MultimodalPrediction> {
    const data = new FormData();
    data.append("image", image);
    data.append("symptoms", symptoms);
    data.append("tabular_json", JSON.stringify(tabular));
    return readJson<MultimodalPrediction>(await fetch(`${baseUrl}/predict/simplified`, { method: "POST", body: data }));
  },
};