export interface Investor {
  id: number;
  name: string;
  age: number;
  aum: number;
  risk_tolerance: string;
  engagement_score: number;
  email_open_rate: number;
  call_frequency: number;
  inactivity_days?: number;
  redemption_intent?: boolean;
  distribution_yield?: number;
  meetings_last_quarter?: number;
  churn_risk_score?: number;
  risk_bucket?: string;
  churn_risk_score_3m?: number;
  churn_risk_score_6m?: number;
  churn_risk_score_12m?: number;
  churn_risk_bucket_3m?: string;
  churn_risk_bucket_6m?: string;
  churn_risk_bucket_12m?: string;
}
