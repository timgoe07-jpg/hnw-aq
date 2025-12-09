export interface Persona {
  id: string;
  name: string;
  short_label: string;
  age_range?: string;
  wealth_range_or_structure?: string;
  primary_goal?: string;
  key_concern?: string;
  investment_behaviour?: string;
  why_private_credit_appeals: string[];
  raw_text?: string;
}
