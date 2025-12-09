export interface ExperienceRole {
  title: string;
  company: string;
  description?: string;
  start_date?: string;
  end_date?: string;
}

export interface Profile {
  full_name: string;
  headline?: string;
  current_title?: string;
  current_company?: string;
  profile_url?: string;
  location?: string;
  industry?: string;
  about_summary?: string;
  experience?: ExperienceRole[];
  follower_count?: number;
  connections?: number;
  photo_url?: string;
  is_premium?: boolean;
}
