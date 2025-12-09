import { Injectable } from "@angular/core";
import { HttpClient, HttpParams } from "@angular/common/http";
import { Observable } from "rxjs";

import { Investor } from "../models/investor.model";
import { Loan } from "../models/loan.model";
import { PaginatedResponse } from "../models/paginated.model";
import { ChurnPrediction, DefaultPrediction } from "../models/prediction.model";
import { ReportResponse } from "../models/report.model";
import { Persona } from "../models/persona.model";
import { Profile } from "../models/profile.model";
import { CaseStudy } from "../models/case-study.model";
import { MatchResponse } from "../models/match-result.model";

@Injectable({ providedIn: "root" })
export class ApiService {
  private baseUrl = "http://localhost:5000/api";

  constructor(private http: HttpClient) {}

  getInvestors(page = 1, pageSize = 100): Observable<PaginatedResponse<Investor>> {
    const params = new HttpParams().set("page", page).set("page_size", pageSize);
    return this.http.get<PaginatedResponse<Investor>>(`${this.baseUrl}/investors`, { params });
  }

  getLoans(page = 1, pageSize = 100): Observable<PaginatedResponse<Loan>> {
    const params = new HttpParams().set("page", page).set("page_size", pageSize);
    return this.http.get<PaginatedResponse<Loan>>(`${this.baseUrl}/loans`, { params });
  }

  predictInvestorChurn(payload: Partial<Investor>): Observable<ChurnPrediction> {
    return this.http.post<ChurnPrediction>(`${this.baseUrl}/predict/investor_churn`, payload);
  }

  predictLoanDefault(payload: Partial<Loan>): Observable<DefaultPrediction> {
    return this.http.post<DefaultPrediction>(`${this.baseUrl}/predict/loan_default`, payload);
  }

  batchRefresh(): Observable<{ investors_updated: number; loans_updated: number; timestamp: string }> {
    return this.http.post<{ investors_updated: number; loans_updated: number; timestamp: string }>(
      `${this.baseUrl}/predict/batch_refresh`,
      {}
    );
  }

  generateReport(params?: { segment_type?: string; segment_value?: string; format?: string }): Observable<ReportResponse> {
    let httpParams = new HttpParams();
    if (params?.segment_type) httpParams = httpParams.set("segment_type", params.segment_type);
    if (params?.segment_value) httpParams = httpParams.set("segment_value", params.segment_value);
    if (params?.format) httpParams = httpParams.set("format", params.format);
    return this.http.post<ReportResponse>(`${this.baseUrl}/report/daily`, {}, { params: httpParams });
  }

  fetchReportPdf(params?: { segment_type?: string; segment_value?: string; format?: string }): Observable<string> {
    let httpParams = new HttpParams();
    if (params?.segment_type) httpParams = httpParams.set("segment_type", params.segment_type);
    if (params?.segment_value) httpParams = httpParams.set("segment_value", params.segment_value);
    if (params?.format) httpParams = httpParams.set("format", params.format);
    return this.http.post(`${this.baseUrl}/report/pdf`, {}, { params: httpParams, responseType: "text" });
  }

  getAnalyticsOverview(): Observable<any> {
    return this.http.get<any>(`${this.baseUrl}/analytics/overview`);
  }

  getAnalyticsSamples(limit = 20): Observable<any> {
    const params = new HttpParams().set("limit", limit);
    return this.http.get<any>(`${this.baseUrl}/analytics/samples`, { params });
  }

  runScenarioPredict(payload: {
    entity_type: "investor" | "loan";
    entity_id: number;
    overrides: Record<string, any>;
    horizon?: string;
  }): Observable<any> {
    return this.http.post(`${this.baseUrl}/scenario/predict`, payload);
  }

  runPortfolioScenario(payload?: { horizon?: string }): Observable<any> {
    return this.http.post(`${this.baseUrl}/scenario/portfolio`, payload || {});
  }

  getTimeline(days = 30): Observable<{ dates: string[]; churn: number[]; default: number[] }> {
    const params = new HttpParams().set("days", days);
    return this.http.get<{ dates: string[]; churn: number[]; default: number[] }>(`${this.baseUrl}/analytics/timeline`, {
      params
    });
  }

  getModelMetrics(): Observable<any> {
    return this.http.get<any>(`${this.baseUrl}/metrics`);
  }

  getAllModelMetrics(): Observable<any> {
    return this.http.get<any>(`${this.baseUrl}/models/metrics`);
  }

  getModelThresholds(problem = "churn", horizon = "6m", family = "ensemble"): Observable<any> {
    const params = new HttpParams().set("problem", problem).set("horizon", horizon).set("family", family);
    return this.http.get<any>(`${this.baseUrl}/models/thresholds`, { params });
  }

  getDrift(): Observable<any> {
    return this.http.get<any>(`${this.baseUrl}/monitoring/drift`);
  }

  getEdaSummary(): Observable<any> {
    return this.http.get<any>(`${this.baseUrl}/eda/summary`);
  }

  getDataAudit(): Observable<any> {
    return this.http.get<any>(`${this.baseUrl}/data/audit`);
  }

  getAlerts(slaDays = 5): Observable<any> {
    const params = new HttpParams().set("sla_days", slaDays);
    return this.http.get<any>(`${this.baseUrl}/alerts`, { params });
  }

  askAi(question: string, focus?: string, context?: any, history: any[] = []): Observable<{ answer: string }> {
    return this.http.post<{ answer: string }>(`${this.baseUrl}/ai/explain`, { question, focus, context, history });
  }

  getInterventions(): Observable<any[]> {
    return this.http.get<any[]>(`${this.baseUrl}/interventions`);
  }

  addIntervention(payload: any): Observable<any[]> {
    return this.http.post<any[]>(`${this.baseUrl}/interventions`, payload);
  }

  // Legacy persona / case study / matching endpoints
  getPersonas(): Observable<Persona[]> {
    return this.http.get<Persona[]>(`${this.baseUrl}/personas`);
  }

  getCaseStudies(): Observable<CaseStudy[]> {
    return this.http.get<CaseStudy[]>(`${this.baseUrl}/case-studies`);
  }

  matchProfiles(payload: { profiles: Profile[] }): Observable<MatchResponse> {
    return this.http.post<MatchResponse>(`${this.baseUrl}/profiles/match`, payload);
  }

  matchProfilesAi(payload: { profiles: Profile[] }): Observable<MatchResponse> {
    return this.http.post<MatchResponse>(`${this.baseUrl}/profiles/ai-match`, payload);
  }

  getSuggestedProfiles(payload: any): Observable<MatchResponse> {
    return this.http.post<MatchResponse>(`${this.baseUrl}/match/suggest-profiles`, payload, {
      withCredentials: true,
    });
  }

  startSuggestedProfilesJob(payload: any): Observable<{ job_id: string }> {
    return this.http.post<{ job_id: string }>(`${this.baseUrl}/match/suggest-profiles/async`, payload, {
      withCredentials: true,
    });
  }

  getSuggestedProfilesJob(jobId: string): Observable<{
    status: string;
    result?: MatchResponse;
    error?: string;
  }> {
    return this.http.get<{ status: string; result?: MatchResponse; error?: string }>(
      `${this.baseUrl}/match/suggest-profiles/job/${jobId}`,
      { withCredentials: true }
    );
  }

  getAiExplanation(profile: Profile): Observable<{ explanation: string }> {
    return this.http.post<{ explanation: string }>(
      `${this.baseUrl}/profiles/ai-explain`,
      { profile },
      { withCredentials: true }
    );
  }

  chatWithProspect(payload: {
    profile: Profile;
    question: string;
    history?: { role: string; content: string }[];
  }): Observable<{ answer: string; history: { role: string; content: string }[] }> {
    return this.http.post<{ answer: string; history: { role: string; content: string }[] }>(
      `${this.baseUrl}/match/prospect-chat`,
      payload,
      { withCredentials: true }
    );
  }

  getLinkedInAuthUrl(): Observable<{ url: string }> {
    return this.http.get<{ url: string }>(`${this.baseUrl}/auth/linkedin/login`, {
      withCredentials: true,
    });
  }

  handleLinkedInCallback(code: string, state: string): Observable<any> {
    return this.http.get(`${this.baseUrl}/auth/linkedin/callback`, {
      params: { code, state },
      withCredentials: true,
    });
  }
}
