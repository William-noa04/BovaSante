export interface TabularInput {
  Breed: string; Region: string; Country: string; Climate_Zone: string; Management_System: string;
  Lactation_Stage: string; Feed_Type: string; Season: string;
  FMD_Vaccine: 0 | 1; Brucellosis_Vaccine: 0 | 1; HS_Vaccine: 0 | 1; BQ_Vaccine: 0 | 1;
  Anthrax_Vaccine: 0 | 1; IBR_Vaccine: 0 | 1; BVD_Vaccine: 0 | 1; Rabies_Vaccine: 0 | 1;
  Age_Months: number; Weight_kg: number; Parity: number; Days_in_Milk: number; Feed_Quantity_kg: number;
  Water_Intake_L: number; Walking_Distance_km: number; Grazing_Duration_hrs: number; Rumination_Time_hrs: number;
  Resting_Hours: number; Body_Temperature_C: number; Heart_Rate_bpm: number; Respiratory_Rate: number;
  Ambient_Temperature_C: number; Humidity_percent: number; Housing_Score: number; Milk_Yield_L: number;
  Previous_Week_Avg_Yield: number; Body_Condition_Score: number; Milking_Interval_hrs: number;
}
export type SimplifiedTabularInput = {
  Age_Months: number;
  Country: string;
  Region: string;
  FMD_Vaccine: number;
  Brucellosis_Vaccine: number;
  HS_Vaccine: number;
  BQ_Vaccine: number;
  Anthrax_Vaccine: number;
  IBR_Vaccine: number;
  BVD_Vaccine: number;
  Rabies_Vaccine: number;
};

export interface ClassProbability { label: string; probability: number }
export interface MultimodalPrediction { predicted_class: string; confidence: number; probabilities: ClassProbability[]; warning: string | null }
export interface HealthResponse { status: string; multimodal_model_loaded: boolean; tabular_only_model_loaded: boolean }

export interface StoredAnalysis { id: string; createdAt: string; cattleId: string; result: MultimodalPrediction }
