import { Component, OnInit } from "@angular/core";
import { FormBuilder, Validators } from "@angular/forms";
import { MatSnackBar } from "@angular/material/snack-bar";

import { ApiService } from "../../services/api.service";
import { ChurnPrediction, DefaultPrediction } from "../../models/prediction.model";
import { ChartConfiguration, ChartOptions } from "chart.js";
import { Investor } from "../../models/investor.model";
import { Loan } from "../../models/loan.model";

@Component({
  selector: "app-predictions",
  templateUrl: "./predictions.component.html",
  styleUrls: ["./predictions.component.scss"]
})
export class PredictionsComponent implements OnInit {
  riskOptions = ["low", "medium", "high"];
  sectorOptions = ["property", "infrastructure", "healthcare", "technology", "hospitality"];

  investorResult?: ChurnPrediction;
  loanResult?: DefaultPrediction;
  investorHorizonRows: { horizon: string; probability: number; bucket: string }[] = [];
  loanHorizonRows: { horizon: string; probability: number; bucket: string }[] = [];
  investorModelRows: { family: string; probability: number; bucket: string }[] = [];
  loanModelRows: { family: string; probability: number; bucket: string }[] = [];
  investorLocalKn?: { probability: number; bucket?: string; k?: number };
  loanLocalKn?: { probability: number; bucket?: string; k?: number };
  investorNeighbors: any[] = [];
  loanNeighbors: any[] = [];
  investorDrivers: any[] = [];
  loanDrivers: any[] = [];
  scenarioInvestorResult?: any;
  scenarioLoanResult?: any;
  scenarioHorizonOptions = ["3m", "6m", "12m"];
  sampleInvestors: Investor[] = [];
  sampleLoans: Loan[] = [];
  loadingInvestor = false;
  loadingLoan = false;
  metrics: any;
  baselineCI: any;
  churnImportanceData: ChartConfiguration<"bar">["data"] = { labels: [], datasets: [{ data: [], backgroundColor: "#0ea5e9" }] };
  defaultImportanceData: ChartConfiguration<"bar">["data"] = { labels: [], datasets: [{ data: [], backgroundColor: "#f59e0b" }] };
  churnCalibrationData: ChartConfiguration<"line">["data"] = { labels: [], datasets: [] };
  defaultCalibrationData: ChartConfiguration<"line">["data"] = { labels: [], datasets: [] };
  investorDriverChart: ChartConfiguration<"bar">["data"] = { labels: [], datasets: [] };
  loanDriverChart: ChartConfiguration<"bar">["data"] = { labels: [], datasets: [] };
  loanContribChart: ChartConfiguration<"bar">["data"] = { labels: [], datasets: [] };
  investorContribChart: ChartConfiguration<"bar">["data"] = { labels: [], datasets: [] };
  chartOptions: ChartOptions = {
    responsive: true,
    plugins: { legend: { display: false } },
    scales: { x: { ticks: { color: "#9bb0c9" } }, y: { ticks: { color: "#9bb0c9" }, beginAtZero: true } }
  };
  lineOptions: ChartOptions<"line"> = {
    responsive: true,
    plugins: { legend: { display: true, labels: { color: "#9bb0c9" } } },
    elements: { point: { radius: 3 } },
    parsing: false,
    scales: {
      x: { type: "linear", ticks: { color: "#9bb0c9" }, grid: { color: "rgba(255,255,255,0.08)" } },
      y: { ticks: { color: "#9bb0c9" }, grid: { color: "rgba(255,255,255,0.08)" }, beginAtZero: true }
    }
  };
  thresholdGrid: any[] = [];
  selectedThreshold = 0.5;
  thresholdSummary?: { threshold: number; tpr: number; fpr: number; precision: number; recall: number; positive_rate: number };
  thresholdMeta = { problem: "churn", horizon: "6m", family: "ensemble" };
  thresholdFamilies = ["ensemble", "adaboost", "logistic"];
  thresholdProblems: Array<"churn" | "default"> = ["churn", "default"];
  thresholdHorizons: Array<"3m" | "6m" | "12m"> = ["3m", "6m", "12m"];
  modelMetricsAll: any = {};
  compareConfig = { problem: "churn", horizon: "6m", family: "ensemble" };
  compareFamilies: string[] = [];
  aiQuestion = "";
  aiAnswer = "";
  aiLoading = false;
  aiHistory: { role: string; content: string }[] = [];
  aiOpen = true;
  cardExplain: Record<string, { text: string; loading: boolean }> = {};
  explainPrompts: Record<string, string> = {
    threshold: "Explain the threshold tuning grid and how precision/recall/TPR/FPR change with the selected threshold.",
    churnImportance: "Explain the churn feature importance chart and what the top features mean.",
    defaultImportance: "Explain the default feature importance chart and what the top features mean.",
    churnCalibration: "Explain the churn calibration curve and how to read alignment between predicted and observed.",
    defaultCalibration: "Explain the default calibration curve and how to read alignment between predicted and observed.",
    whatIf: "Explain the polynomial what-if sliders and how the curves show non-linear relationships.",
    surfaces: "Explain the 2D scenario surface slices and how to interpret the lines.",
    compare: "Explain the compare metrics table and what AUC/precision/recall values imply.",
    thresholdsMeta: "Explain the problem/horizon/family selections for thresholds and why they matter.",
    churnForm: "Explain the investor churn input form and what the model outputs represent.",
    loanForm: "Explain the loan default input form and what the model outputs represent.",
  };
  whatIfChurn = 60;
  whatIfDefault = 0.6;
  whatIfCurves: { churn?: ChartConfiguration<"line">["data"]; default?: ChartConfiguration<"line">["data"] } = {};
  surfaceCurves: { churn?: ChartConfiguration<"line">["data"]; default?: ChartConfiguration<"line">["data"] } = {};

  investorForm = this.fb.nonNullable.group({
    age: [45, [Validators.required]],
    aum: [1500000, [Validators.required]],
    risk_tolerance: ["medium", Validators.required],
    engagement_score: [60, [Validators.required]],
    email_open_rate: [0.5, [Validators.required]],
    call_frequency: [4, [Validators.required]],
    inactivity_days: [30, [Validators.required]],
    redemption_intent: [false, [Validators.required]],
    distribution_yield: [0.07, [Validators.required]],
    meetings_last_quarter: [2, [Validators.required]],
  });

  investorScenarioForm = this.fb.nonNullable.group({
    entity_id: [0, [Validators.required]],
    horizon: ["6m", [Validators.required]],
    engagement_score: [60, [Validators.required]],
    inactivity_days: [30, [Validators.required]],
    call_frequency: [4, [Validators.required]],
  });

  loanForm = this.fb.nonNullable.group({
    amount: [750000, [Validators.required]],
    ltv_ratio: [0.6, [Validators.required]],
    term_months: [36, [Validators.required]],
    sector: ["property", Validators.required],
    arrears_flag: [false, Validators.required],
    dscr: [1.2, [Validators.required]],
    covenants_flag: [false, Validators.required],
    collateral_score: [0.7, [Validators.required]],
  });

  loanScenarioForm = this.fb.nonNullable.group({
    entity_id: [0, [Validators.required]],
    horizon: ["6m", [Validators.required]],
    ltv_ratio: [0.6, [Validators.required]],
    dscr: [1.2, [Validators.required]],
    term_months: [36, [Validators.required]],
  });

  constructor(private fb: FormBuilder, private api: ApiService, private snackBar: MatSnackBar) {}

  ngOnInit(): void {
    this.loadMetrics();
    this.loadAllModelMetrics();
    this.loadSamples();
    this.loadThresholds();
  }

  submitInvestor(): void {
    if (this.investorForm.invalid) {
      this.investorForm.markAllAsTouched();
      return;
    }
    this.loadingInvestor = true;
    const payload = { ...this.investorForm.value };
    payload.age = Number(payload.age);
    payload.aum = Number(payload.aum);
    payload.engagement_score = Number(payload.engagement_score);
    payload.email_open_rate = Number(payload.email_open_rate);
    payload.call_frequency = Number(payload.call_frequency);

    this.api.predictInvestorChurn(payload).subscribe({
      next: (res) => {
        this.investorResult = res;
        this.investorHorizonRows = this.mapHorizons(res.horizons);
        this.investorModelRows = this.toModelRows(res.models, "6m");
        this.investorLocalKn = res.local_knn?.["6m"] || undefined;
        this.investorNeighbors = res.neighbors || [];
        this.investorDrivers = res.drivers || [];
        this.investorDriverChart = this.toDriverChart(this.investorDrivers, "Churn drivers");
        this.loadingInvestor = false;
      },
      error: (err) => this.handleError("Unable to predict churn", err, "investor")
    });
  }

  submitLoan(): void {
    if (this.loanForm.invalid) {
      this.loanForm.markAllAsTouched();
      return;
    }
    this.loadingLoan = true;
    const payload = { ...this.loanForm.value };
    payload.amount = Number(payload.amount);
    payload.ltv_ratio = Number(payload.ltv_ratio);
    payload.term_months = Number(payload.term_months);
    payload.arrears_flag = !!payload.arrears_flag;

    this.api.predictLoanDefault(payload).subscribe({
      next: (res) => {
        this.loanResult = res;
        this.loanHorizonRows = this.mapHorizons(res.horizons);
        this.loanModelRows = this.toModelRows(res.models, "6m");
        this.loanLocalKn = res.local_knn?.["6m"] || undefined;
        this.loanNeighbors = res.neighbors || [];
        this.loanDrivers = res.drivers || [];
        this.loanDriverChart = this.toDriverChart(this.loanDrivers, "Default drivers");
        this.loadingLoan = false;
      },
      error: (err) => this.handleError("Unable to predict default risk", err, "loan")
    });
  }

  formatPercent(prob?: number): string {
    if (prob === undefined || prob === null) return "0.0%";
    return `${(prob * 100).toFixed(1)}%`;
  }

  private handleError(message: string, err: any, type: "investor" | "loan"): void {
    console.error(message, err);
    if (type === "investor") this.loadingInvestor = false;
    if (type === "loan") this.loadingLoan = false;
    this.snackBar.open(message, "Close", { duration: 3000 });
  }

  private loadMetrics(): void {
    this.api.getModelMetrics().subscribe({
      next: (m) => {
        this.metrics = m;
        this.baselineCI = m?.bootstrap_ci;
        const horizon = this.thresholdMeta.horizon as "3m" | "6m" | "12m";
        const churnBlock = (m?.churn?.[horizon]) || m?.churn;
        const defaultBlock = (m?.default?.[horizon]) || m?.default;
        this.churnImportanceData = this.toBarData(churnBlock?.feature_importance, "#0ea5e9");
        this.defaultImportanceData = this.toBarData(defaultBlock?.feature_importance, "#f59e0b");
        this.churnCalibrationData = this.toCalibrationData(churnBlock?.calibration, "Churn");
        this.defaultCalibrationData = this.toCalibrationData(defaultBlock?.calibration, "Default");
        this.buildContributionCharts();
      },
      error: () => {}
    });
  }

  private loadAllModelMetrics(): void {
    this.api.getAllModelMetrics().subscribe({
      next: (resp) => {
        this.modelMetricsAll = resp || {};
        this.updateCompareFamilies();
        this.whatIfCurves = this.buildWhatIfCurves();
        this.surfaceCurves = this.buildSurfaceCurves();
      },
      error: (err) => console.warn("Unable to load model metrics", err)
    });
  }

  get compareMetrics(): any {
    const problemBlock = this.modelMetricsAll?.[this.compareConfig.problem];
    if (!problemBlock) return null;
    const familyBlock = problemBlock?.[this.compareConfig.family] || problemBlock;
    const horizonBlock = familyBlock?.[this.compareConfig.horizon] || familyBlock;
    return horizonBlock || null;
  }

  updateCompareFamilies(): void {
    const problemBlock = this.modelMetricsAll?.[this.compareConfig.problem] || {};
    const keys = Object.keys(problemBlock || {}).filter((k) => typeof problemBlock[k] === "object");
    this.compareFamilies = keys.length ? keys : ["ensemble"];
    if (!this.compareFamilies.includes(this.compareConfig.family)) {
      this.compareConfig.family = this.compareFamilies[0];
    }
  }

  onCompareProblemChange(): void {
    this.updateCompareFamilies();
  }

  private loadThresholds(problem: "churn" | "default" = "churn", horizon: "3m" | "6m" | "12m" = "6m", family = "ensemble") {
    this.thresholdMeta = { problem, horizon, family };
    this.api.getModelThresholds(problem, horizon, family).subscribe({
      next: (resp) => {
        this.thresholdGrid = resp?.thresholds || [];
        this.selectedThreshold = 0.5;
        this.updateThresholdSummary();
      },
      error: (err) => console.warn("Unable to load thresholds", err)
    });
  }

  private toBarData(imp: Record<string, number> | undefined, color: string) {
    if (!imp) return { labels: [], datasets: [] };
    const entries = Object.entries(imp);
    return { labels: entries.map((e) => e[0]), datasets: [{ data: entries.map((e) => e[1]), backgroundColor: color }] };
  }

  private toCalibrationData(cal: any, label: string) {
    if (!cal?.fraction_positives || !cal?.mean_predictions) return { labels: [], datasets: [] };
    return {
      labels: cal.mean_predictions.map((_: any, idx: number) => `Bin ${idx + 1}`),
      datasets: [
        {
          label,
          data: cal.fraction_positives,
          borderColor: "#19c3b1",
          backgroundColor: "rgba(25,195,177,0.2)",
          tension: 0.25,
          fill: true
        }
      ]
    };
  }

  private toDriverChart(drivers: any[], label: string) {
    if (!drivers?.length) return { labels: [], datasets: [] };
    const labels = drivers.map((d) => d.feature);
    const deltas = drivers.map((d) => (d.delta_pct || 0) * 100);
    const colors = deltas.map((d) => (d >= 0 ? "rgba(239,68,68,0.5)" : "rgba(34,197,94,0.5)"));
    return {
      labels,
      datasets: [
        {
          label,
          data: deltas,
          backgroundColor: colors,
          borderColor: colors.map((c) => c.replace("0.5", "1")),
          borderWidth: 1,
        }
      ]
    };
  }

  private buildContributionCharts(): void {
    const contrib = (this.metrics as any)?.contributions || {};
    const invMap = contrib.churn || {};
    const loanMap = contrib.default || {};
    const invFirst = Object.values(invMap)[0] as any;
    if (invFirst) {
      const labels = Object.keys(invFirst);
      const data = labels.map((k) => invFirst[k]);
      this.investorContribChart = { labels, datasets: [{ label: "Logistic contribution", data, backgroundColor: "#0ea5e9" }] };
    }
    const loanFirst = Object.values(loanMap)[0] as any;
    if (loanFirst) {
      const labels = Object.keys(loanFirst);
      const data = labels.map((k) => loanFirst[k]);
      this.loanContribChart = { labels, datasets: [{ label: "Logistic contribution", data, backgroundColor: "#f59e0b" }] };
    }
  }

  private buildWhatIfCurves() {
    const nl = this.modelMetricsAll?.non_linear || {};
    const make = (series: any, label: string, color: string, pointX: number) => {
      if (!series?.x || !series?.y) return undefined;
      const pointY = this.interpolate(series.x, series.y, pointX);
      return {
        datasets: [
          { data: series.x.map((xVal: number, idx: number) => ({ x: xVal, y: series.y[idx] })), label, borderColor: color, backgroundColor: "rgba(0,0,0,0)", tension: 0.15, fill: false },
          { data: [{ x: pointX, y: pointY }], label: "Scenario", pointRadius: 6, showLine: false, borderColor: "#f59e0b", backgroundColor: "#f59e0b" }
        ],
      };
    };
    return {
      churn: make(nl.churn?.engagement_score, "Churn vs engagement", "#22c55e", this.whatIfChurn),
      default: make(nl.default?.ltv_ratio, "Default vs LTV", "#ef4444", this.whatIfDefault),
    };
  }

  private buildSurfaceCurves() {
    const surf = this.modelMetricsAll?.surfaces || {};
    const make = (surface: any, color: string) => {
      if (!surface?.x || !surface?.y || !surface?.z) return undefined;
      const datasets = (surface.y as number[]).map((yVal: number, idx: number) => ({
        label: `${surface.y_label}=${yVal.toFixed(1)}`,
        data: (surface.x as number[]).map((xVal: number, j: number) => ({ x: xVal, y: surface.z[idx][j] })),
        borderColor: color,
        backgroundColor: "rgba(0,0,0,0)",
        tension: 0.15,
        fill: false,
      }));
      return { datasets };
    };
    return {
      churn: make(surf.churn, "#0ea5e9"),
      default: make(surf.default, "#f59e0b"),
    };
  }

  private interpolate(xs: number[], ys: number[], x: number): number {
    if (!xs?.length || !ys?.length) return 0;
    for (let i = 0; i < xs.length - 1; i++) {
      const x1 = xs[i];
      const x2 = xs[i + 1];
      if (x >= x1 && x <= x2) {
        const y1 = ys[i];
        const y2 = ys[i + 1];
        const t = (x - x1) / (x2 - x1 || 1);
        return y1 + t * (y2 - y1);
      }
    }
    return ys[ys.length - 1];
  }

  askAi(): void {
    if (!this.aiQuestion) return;
    this.aiLoading = true;
    this.aiHistory.push({ role: "user", content: this.aiQuestion });
    const context = {
      investorResult: this.investorResult,
      loanResult: this.loanResult,
      investorForm: this.investorForm.value,
      loanForm: this.loanForm.value,
      metrics: this.metrics,
    };
    this.api.askAi(this.aiQuestion, "predictions", context, this.aiHistory).subscribe({
      next: (resp) => {
        this.aiAnswer = resp.answer;
        this.aiHistory.push({ role: "assistant", content: resp.answer });
        this.aiLoading = false;
        this.aiQuestion = "";
      },
      error: (err) => this.handleError("AI explanation failed", err, "investor")
    });
  }

  explainCard(id: string, prompt: string, context: any = {}): void {
    const curr = this.cardExplain[id] || { text: "", loading: false };
    this.cardExplain[id] = { ...curr, loading: true };
    this.api.askAi(prompt, "predictions-card", context, []).subscribe({
      next: (resp) => (this.cardExplain[id] = { text: resp.answer, loading: false }),
      error: () => {
        this.cardExplain[id] = { text: "Unable to fetch explanation right now.", loading: false };
        this.snackBar.open("Explain failed", "Close", { duration: 2000 });
      }
    });
  }

  loadSamples(): void {
    this.api.getAnalyticsSamples(50).subscribe({
      next: (resp) => {
        this.sampleInvestors = resp.investors || [];
        this.sampleLoans = resp.loans || [];
        if (this.sampleInvestors.length) {
          this.investorScenarioForm.patchValue({
            entity_id: this.sampleInvestors[0].id,
            engagement_score: this.sampleInvestors[0].engagement_score ?? 60,
            inactivity_days: (this.sampleInvestors[0] as any).inactivity_days ?? 30,
            call_frequency: this.sampleInvestors[0].call_frequency ?? 4
          });
        }
        if (this.sampleLoans.length) {
          this.loanScenarioForm.patchValue({
            entity_id: this.sampleLoans[0].id,
            ltv_ratio: this.sampleLoans[0].ltv_ratio ?? 0.6,
            dscr: (this.sampleLoans[0] as any).dscr ?? 1.2,
            term_months: this.sampleLoans[0].term_months ?? 36
          });
        }
      },
      error: (err) => console.warn("Unable to load samples", err)
    });
  }

  onInvestorSelect(id: number): void {
    const found = this.sampleInvestors.find((i) => i.id === Number(id));
    if (found) {
      this.investorScenarioForm.patchValue({
        engagement_score: found.engagement_score ?? 60,
        inactivity_days: (found as any).inactivity_days ?? 30,
        call_frequency: found.call_frequency ?? 4
      });
    }
  }

  onLoanSelect(id: number): void {
    const found = this.sampleLoans.find((l) => l.id === Number(id));
    if (found) {
      this.loanScenarioForm.patchValue({
        ltv_ratio: found.ltv_ratio ?? 0.6,
        dscr: (found as any).dscr ?? 1.2,
        term_months: found.term_months ?? 36
      });
    }
  }

  runInvestorScenario(): void {
    if (this.investorScenarioForm.invalid) {
      this.investorScenarioForm.markAllAsTouched();
      return;
    }
    const payload = this.investorScenarioForm.getRawValue();
    this.api
      .runScenarioPredict({
        entity_type: "investor",
        entity_id: Number(payload.entity_id),
        horizon: payload.horizon,
        overrides: {
          engagement_score: Number(payload.engagement_score),
          inactivity_days: Number(payload.inactivity_days),
          call_frequency: Number(payload.call_frequency)
        }
      })
      .subscribe({
        next: (res) => (this.scenarioInvestorResult = res),
        error: (err) => this.handleError("Scenario run failed", err, "investor")
      });
  }

  runLoanScenario(): void {
    if (this.loanScenarioForm.invalid) {
      this.loanScenarioForm.markAllAsTouched();
      return;
    }
    const payload = this.loanScenarioForm.getRawValue();
    this.api
      .runScenarioPredict({
        entity_type: "loan",
        entity_id: Number(payload.entity_id),
        horizon: payload.horizon,
        overrides: {
          ltv_ratio: Number(payload.ltv_ratio),
          dscr: Number(payload.dscr),
          term_months: Number(payload.term_months)
        }
      })
      .subscribe({
        next: (res) => (this.scenarioLoanResult = res),
        error: (err) => this.handleError("Scenario run failed", err, "loan")
      });
  }

  deltaLabel(base?: number, scenario?: number): string {
    if (base === undefined || scenario === undefined) return "0.0%";
    const delta = (scenario - base) * 100;
    const sign = delta > 0 ? "+" : "";
    return `${sign}${delta.toFixed(1)}%`;
  }

  private mapHorizons(h?: Record<string, { probability: number; bucket: string }>) {
    if (!h) return [];
    return Object.entries(h).map(([key, val]) => ({ horizon: key, probability: val.probability, bucket: val.bucket }));
  }

  private toModelRows(models: Record<string, Record<string, { probability: number; bucket: string }>> | undefined, horizon: string) {
    if (!models) return [];
    const rows: { family: string; probability: number; bucket: string }[] = [];
    Object.entries(models).forEach(([family, horizons]) => {
      const entry = (horizons as any)[horizon] || Object.values(horizons || {})[0];
      if (entry) {
        rows.push({ family, probability: entry.probability, bucket: entry.bucket });
      }
    });
    return rows.sort((a, b) => a.family.localeCompare(b.family));
  }

  updateThresholdSummary(): void {
    if (!this.thresholdGrid?.length) {
      this.thresholdSummary = undefined;
      return;
    }
    const closest = this.thresholdGrid.reduce((prev: any, curr: any) =>
      Math.abs(curr.threshold - this.selectedThreshold) < Math.abs((prev?.threshold ?? 0) - this.selectedThreshold) ? curr : prev
    );
    this.thresholdSummary = closest;
  }

  onThresholdChange(val: number | { value: number | null } | null): void {
    const next = typeof val === "number" ? val : val?.value ?? null;
    if (next === null || next === undefined) return;
    this.selectedThreshold = next;
    this.updateThresholdSummary();
  }

  onThresholdMetaChange(): void {
    this.loadThresholds(this.thresholdMeta.problem as "churn" | "default", this.thresholdMeta.horizon as "3m" | "6m" | "12m", this.thresholdMeta.family);
    // Refresh horizon-specific charts for importance/calibration
    this.loadMetrics();
  }

  onWhatIfChurnChange(val: number | { value: number | null }): void {
    const next = typeof val === "number" ? val : val?.value ?? null;
    if (next === null || next === undefined) return;
    this.whatIfChurn = next;
    this.whatIfCurves = this.buildWhatIfCurves();
  }

  onWhatIfDefaultChange(val: number | { value: number | null }): void {
    const next = typeof val === "number" ? val : val?.value ?? null;
    if (next === null || next === undefined) return;
    this.whatIfDefault = next;
    this.whatIfCurves = this.buildWhatIfCurves();
  }
}
