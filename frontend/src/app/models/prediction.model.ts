export interface ChurnPrediction {
  churn_probability: number;
  risk_bucket: string;
  explanation: string;
  horizons?: Record<string, { probability: number; bucket: string }>;
  models?: Record<string, Record<string, { probability: number; bucket: string }>>;
  primary_family?: string;
  local_knn?: Record<string, { probability: number; bucket?: string; k?: number }>;
  neighbors?: Array<{ id: number; label: number; distance: number; engagement_score?: number; inactivity_days?: number; email_open_rate?: number }>;
  drivers?: Array<{ feature: string; value: number; portfolio_median: number; delta_pct: number }>;
}

export interface DefaultPrediction {
  default_probability: number;
  risk_bucket: string;
  explanation: string;
  horizons?: Record<string, { probability: number; bucket: string }>;
  models?: Record<string, Record<string, { probability: number; bucket: string }>>;
  primary_family?: string;
  local_knn?: Record<string, { probability: number; bucket?: string; k?: number }>;
  neighbors?: Array<{ id: number; label: number; distance: number; sector?: string; ltv_ratio?: number; dscr?: number }>;
  drivers?: Array<{ feature: string; value: number; portfolio_median: number; delta_pct: number }>;
}
