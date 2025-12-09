import { Profile } from "./profile.model";

export interface PersonaMatchScore {
  persona_id: string;
  persona_name: string;
  score: number;
  reason?: string;
}

export interface ProfileMatchResult {
  profile: Profile;
  matches: PersonaMatchScore[];
}

export interface MatchResponse {
  results?: ProfileMatchResult[];
  profiles?: ProfileMatchResult[];
  search_plan?: {
    keywords: string[];
    explanation?: string;
    raw_query?: string;
    attempted_queries?: {
      query: string;
      filters?: {
        location?: string;
        industry?: string;
      };
    }[];
  };
  metadata?: {
    used_ai_ranking?: boolean;
    synthetic_profiles?: boolean;
  };
}
