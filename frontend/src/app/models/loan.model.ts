export interface Loan {
  id: number;
  investor_id: number;
  amount: number;
  ltv_ratio: number;
  term_months: number;
  sector: string;
  arrears_flag: boolean;
  dscr?: number;
  covenants_flag?: boolean;
  collateral_score?: number;
  default_risk_score?: number;
  risk_bucket?: string;
  default_risk_score_3m?: number;
  default_risk_score_6m?: number;
  default_risk_score_12m?: number;
  default_risk_bucket_3m?: string;
  default_risk_bucket_6m?: string;
  default_risk_bucket_12m?: string;
}
