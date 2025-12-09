export interface SummaryKpis {
  investor_buckets: Record<string, number>;
  loan_buckets: Record<string, number>;
  avg_churn_risk: number;
  prev_day_avg_churn_risk: number;
  high_default_exposure: number;
  churn_by_tolerance?: Record<string, number>;
  sector_default?: Record<string, { high: number; exposure: number }>;
  engagement_avg?: number;
  email_open_avg?: number;
  distribution_yield?: Record<string, number>;
  dscr?: Record<string, number>;
  engagement_trend?: number[];
  dscr_trend?: number[];
  deltas?: any;
  playbooks?: any;
  top_investors: { id: number; name: string; probability: number; bucket: string }[];
  top_loans: { id: number; sector: string; amount: number; probability: number; bucket: string }[];
}

export interface ReportResponse {
  generated_at: string;
  summary_kpis: SummaryKpis;
  report_markdown: string;
  base_report_markdown?: string;
}
