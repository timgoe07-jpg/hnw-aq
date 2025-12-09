import { Component, OnInit, ChangeDetectorRef } from "@angular/core";
import { MatSnackBar } from "@angular/material/snack-bar";
import { forkJoin, of } from "rxjs";
import { catchError } from "rxjs/operators";

import { ApiService } from "../../services/api.service";
import { Investor } from "../../models/investor.model";
import { Loan } from "../../models/loan.model";
import { ReportResponse } from "../../models/report.model";
import { ChartOptions, ChartConfiguration } from "chart.js";

@Component({
  selector: "app-dashboard",
  templateUrl: "./dashboard.component.html",
  styleUrls: ["./dashboard.component.scss"]
})
export class DashboardComponent implements OnInit {
  investors: Investor[] = [];
  loans: Loan[] = [];
  baselineInvestors: Investor[] = [];
  baselineLoans: Loan[] = [];
  baselineReport?: ReportResponse;
  baselineAnalytics: any;
  baselineMetrics: any;
  baselineSamples: any;
  baselineTimeline: any;
  report?: ReportResponse;
  Math = Math;
  loading = false;
  charts: {
    churnBuckets: { labels: string[]; data: number[] };
    defaultBuckets: { labels: string[]; data: number[] };
    sectorExposure: { labels: string[]; data: number[] };
    aumBuckets: { labels: string[]; data: number[] };
    yieldEngagementTrend: { labels: string[]; data: number[] };
    dscrSummary: { labels: string[]; data: number[] };
  } = {
    churnBuckets: { labels: [], data: [] },
    defaultBuckets: { labels: [], data: [] },
    sectorExposure: { labels: [], data: [] },
    aumBuckets: { labels: [], data: [] },
    yieldEngagementTrend: { labels: [], data: [] },
    dscrSummary: { labels: [], data: [] }
  };
  timeline = { dates: [] as string[], churn: [] as number[], default: [] as number[] };
  chartOptions: ChartOptions<"bar"> = {
    responsive: true,
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { color: "#9bb0c9" }, grid: { color: "rgba(255,255,255,0.05)" } },
      y: { ticks: { color: "#9bb0c9" }, grid: { color: "rgba(255,255,255,0.05)" }, beginAtZero: true }
    }
  };
  samples: { investors: Investor[]; loans: Loan[] } = { investors: [], loans: [] };
  investorRows: Investor[] = [];
  loanRows: Loan[] = [];
  investorCount = 0;
  loanCount = 0;
  investorFilter = "";
  loanFilter = "";
  investorSortKey: keyof Investor | "" = "";
  investorSortDir: "asc" | "desc" = "asc";
  loanSortKey: keyof Loan | "" = "";
  loanSortDir: "asc" | "desc" = "asc";
  investorPage = 0;
  loanPage = 0;
  pageSize = 5;
  pageSizeOptions = [5, 10, 25];
  aiQuestion = "";
  aiAnswer = "";
  aiLoading = false;
  aiHistory: { role: string; content: string }[] = [];
  selectedHorizon: "3m" | "6m" | "12m" = "6m";
  analyticsRaw: any;
  aiOpen = true;
  modelFamilies = [
    { id: "ensemble", label: "Ensemble (RF+GB)" },
    { id: "rf", label: "Random Forest" },
    { id: "logreg", label: "Logistic Regression" },
    { id: "adaboost", label: "AdaBoost" },
    { id: "bagging", label: "Bagging" },
    { id: "knn", label: "k-NN (local)" }
  ];
  selectedFamily = "ensemble";
  compareFamily = "adaboost";
  modelMetrics: any;
  thresholdValue = 0.5;
  costFP = 1000;
  costFN = 5000;
  thresholdSummary: { churn?: any; default?: any } = {};
  cardExplain: Record<string, { text: string; loading: boolean }> = {};
  explainPrompts: Record<string, string> = {
    investorBuckets: "Explain the churn bucket counts (Low/Medium/High) for investors and how they are derived.",
    loanBuckets: "Explain the default bucket counts (Low/Medium/High) for loans and how they are derived.",
    engagementSnapshot: "Explain the engagement/yield snapshot and how it reflects investor behavior.",
    cohortModels: "Explain the cohort models overview and why segment-level probabilities/buckets matter.",
    cohortRisk: "Explain the cohort risk averages tables and what each slice (tolerance, AUM, sector) shows.",
    segmentRisk: "Explain the segment risk tables and how to interpret averages/buckets per segment.",
    predictorExplorer: "Explain the predictor explorer tables/charts and how single-feature AUC helps prioritize signals.",
    diagChurn: "Explain the 1D diagnostics for churn (logistic vs k-NN) and what the curves mean.",
    diagDefault: "Explain the 1D diagnostics for default (logistic vs k-NN) and what the curves mean.",
    fairness: "Explain the fairness/segment performance charts and why AUC/AP by segment matter.",
    nonLinearChurn: "Explain the non-linear churn pattern chart and what the curve implies.",
    nonLinearDefault: "Explain the non-linear default pattern chart and what the curve implies.",
    correlation: "Explain the collinearity watchlist and why highly correlated features are flagged.",
    imbalanceCard: "Explain the imbalance snapshot and how positive rates/segments affect models.",
    robustness: "Explain the robustness lens (bootstrap CI, sensitivity, segment ROC) and why it matters.",
    rocExplorer: "Explain the ROC explorer scenario points and how they relate to thresholds.",
    edaInvestors: "Explain the investor EDA histograms/outliers and what they show about data distribution.",
    edaLoans: "Explain the loan EDA histograms/outliers and what they show about data distribution.",
    forecaster: "Explain the regression forecaster metrics and coefficients.",
    regressionKpi: "Explain the regression KPI tables and how to read R²/RMSE/MAE.",
    drivers: "Explain the per-entity drivers bar charts and how contributions relate to SHAP-like effects.",
    brokerSector: "Explain the broker vs sector risk table and how to interpret the percentages.",
    selectionRf: "Explain the RF estimator selection chart and how train/val/CV AUC guide tuning.",
    selectionRfDefault: "Explain the RF estimator selection for default and its implications.",
    selectionGb: "Explain the gradient boosting sweep (churn) and how learning rate affects performance.",
    selectionGbDefault: "Explain the gradient boosting sweep (default) and how learning rate affects performance.",
    seedVariability: "Explain the seed variability table and why CV stability matters.",
    hyperparams: "Explain the selected hyperparameters for churn/default models.",
    singleFeature: "Explain the single-feature benchmark bar charts and why they’re useful.",
    segmentRoc: "Explain the segment ROC overlays and what segment AUC means.",
    biasGap: "Explain the bias/disparity charts and how TPR gaps are interpreted.",
    surfaceSlices: "Explain the what-if contour slices and how to read the lines.",
    voting: "Explain the voting score charts and why top features matter.",
    modelComparison: "Explain the model comparison metrics (AUC/precision/recall) and how they relate to churn/default risk for the selected family and horizon.",
    radar: "Explain the radar diagnostics chart and what each spoke indicates about model robustness.",
    buckets: "Explain the bucket distributions and why Low/Med/High buckets matter.",
    riskCockpit: "Explain the SLA risk cockpit table and how to interpret active/breach status.",
    interventions: "Explain the interventions log and how adjustments affect scenarios.",
    familyOverlay: "Explain the ROC/PR overlays across model families and what deltas mean.",
    stability: "Explain the model stability/drift indicators.",
    threshold: "Explain threshold tuning, TPR/FPR/precision/recall trade-offs, and the cost inputs.",
    roc: "Explain the ROC curves shown for churn/default and what good performance looks like.",
    pr: "Explain the precision-recall curves and why they matter for imbalance.",
    portfolio: "Explain the portfolio what-if scenario outputs and how to read base vs scenario deltas.",
    readiness: "Explain the data readiness findings and their impact on model quality.",
    health: "Explain the data health signals and why missingness/encoding warnings matter.",
    report: "Explain what the report preview contains and how it is generated.",
    samplesInvestors: "Explain the sample investor table columns and how to interpret risk buckets.",
    samplesLoans: "Explain the sample loan table columns and how to interpret default buckets.",
    explainability: "Explain the per-entity snapshot and what the listed attributes imply for risk.",
  };
  driftSummary: any;
  dataAudit: any;
  alerts: any = { investors: [], loans: [], summary: {} };
  slaThreshold = 5;
  aiKpiSnapshot: any;
  interventions: any[] = [];
  scenarioResult: any;
  selectedInvestor: Investor | null = null;
  selectedLoan: Loan | null = null;
  explainAnchor!: HTMLElement | null;
  rocData: ChartConfiguration<"line">["data"] = { datasets: [] };
  prData: ChartConfiguration<"line">["data"] = { datasets: [] };
  radarData: ChartConfiguration<"radar">["data"] = { labels: [], datasets: [] };
  bucketData: ChartConfiguration<"bar">["data"] = { labels: ["Low", "Medium", "High"], datasets: [] };
  compareRoc: ChartConfiguration<"line">["data"] = { datasets: [] };
  comparePr: ChartConfiguration<"line">["data"] = { datasets: [] };
  predictorCharts: { churn: ChartConfiguration<"bar">["data"]; default: ChartConfiguration<"bar">["data"] } = {
    churn: { labels: [], datasets: [] },
    default: { labels: [], datasets: [] },
  };
  singleFeatureCharts: { churn?: ChartConfiguration<"bar">["data"]; default?: ChartConfiguration<"bar">["data"] } = {};
  coeffStability: { churn?: any; default?: any } = {};
  imbalanceDetail: any;
  modelComparator: ChartConfiguration<"radar">["data"] = { labels: [], datasets: [] };
  regressionKpi: any;
  diagnosticsCurves: { churn?: ChartConfiguration<"line">["data"]; default?: ChartConfiguration<"line">["data"] } = {};
  selectionCurves: ChartConfiguration<"line">["data"] = { datasets: [] };
  selectionCurvesDefault: ChartConfiguration<"line">["data"] = { datasets: [] };
  selectionCurvesGb: ChartConfiguration<"line">["data"] = { datasets: [] };
  selectionCurvesGbDefault: ChartConfiguration<"line">["data"] = { datasets: [] };
  nonLinearCharts: { churn?: ChartConfiguration<"line">["data"]; default?: ChartConfiguration<"line">["data"] } = {};
  imbalance: any;
  robustness: any;
  edaSummary: any;
  rocExplorer: ChartConfiguration<"line">["data"] = { datasets: [] };
  edaCharts: { investor?: any; loan?: any } = {};
  fairnessCharts: {
    churn?: ChartConfiguration<"bar">["data"];
    churnAp?: ChartConfiguration<"bar">["data"];
    default?: ChartConfiguration<"bar">["data"];
    defAp?: ChartConfiguration<"bar">["data"];
  } = {};
  forecaster: any;
  missingHeatmap: any;
  segmentBias: any;
  segmentRocCharts: { churn?: ChartConfiguration<"bar">["data"]; default?: ChartConfiguration<"bar">["data"] } = {};
  biasGapCharts: { churn?: ChartConfiguration<"bar">["data"]; default?: ChartConfiguration<"bar">["data"] } = {};
  votingCharts: { churn?: ChartConfiguration<"bar">["data"]; default?: ChartConfiguration<"bar">["data"] } = {};
  surfaceSlices: { churn?: ChartConfiguration<"line">["data"]; default?: ChartConfiguration<"line">["data"] } = {};
  dashboardContrib: { investor?: ChartConfiguration<"bar">["data"]; loan?: ChartConfiguration<"bar">["data"] } = {};
  selectedInvestorId: number | null = null;
  selectedLoanId: number | null = null;
  brokerPage = 0;
  brokerPageSize = 6;
  lineOptions: ChartOptions<"line"> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: true, labels: { color: "#9bb0c9" } } },
    elements: { point: { radius: 2 } },
    scales: {
      x: { type: "linear", ticks: { color: "#9bb0c9" }, grid: { color: "rgba(255,255,255,0.08)" } },
      y: { min: 0, max: 1, ticks: { color: "#9bb0c9" }, grid: { color: "rgba(255,255,255,0.08)" } }
    }
  };
  segmentHeatmaps: any = {};
  brokerHeatmaps: any = {};
  surfaceOptions: ChartOptions<"line"> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: true, labels: { color: "#9bb0c9" } } },
    elements: { point: { radius: 0 } },
    scales: {
      x: { type: "linear", ticks: { color: "#9bb0c9" }, grid: { color: "rgba(255,255,255,0.08)" } },
      y: { ticks: { color: "#9bb0c9" }, grid: { color: "rgba(255,255,255,0.08)" } }
    }
  };
  selectionOptions: ChartOptions<"line"> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: true, labels: { color: "#9bb0c9" } } },
    elements: { point: { radius: 3 } },
    scales: {
      x: { type: "linear", ticks: { color: "#9bb0c9" }, grid: { color: "rgba(255,255,255,0.08)" } },
      y: { min: 0, max: 1, ticks: { color: "#9bb0c9" }, grid: { color: "rgba(255,255,255,0.08)" } }
    }
  };

  constructor(
    private api: ApiService,
    private snackBar: MatSnackBar,
    private cdr: ChangeDetectorRef
  ) {}

  ngOnInit(): void {
    this.loadData();
    // Cache the AI explain card element for smooth scroll later
    setTimeout(() => {
      this.explainAnchor = document.getElementById("ai-explain-card");
    });
  }

  loadData(): void {
    this.loading = true;
    this.api.batchRefresh().subscribe({
      next: () => {
        forkJoin({
          investors: this.api.getInvestors(),
          loans: this.api.getLoans(),
          report: this.api.generateReport(),
          analytics: this.api.getAnalyticsOverview(),
          samples: this.api.getAnalyticsSamples(50),
          metrics: this.api.getAllModelMetrics(),
          drift: this.api.getDrift().pipe(catchError(() => of(null))),
          audit: this.api.getDataAudit().pipe(catchError(() => of(null))),
          alerts: this.api.getAlerts().pipe(catchError(() => of(null))),
          timeline: this.api.getTimeline(45).pipe(catchError(() => of({ dates: [], churn: [], default: [] }))),
          interventions: this.api.getInterventions().pipe(catchError(() => of([]))),
          eda: this.api.getEdaSummary().pipe(catchError(() => of(null))),
        }).subscribe({
          next: ({ investors, loans, report, analytics, samples, timeline, metrics, drift, audit, alerts, interventions, eda }) => {
            const incomingInvestors = investors?.items || [];
            const incomingLoans = loans?.items || [];
            const hasInvestors = incomingInvestors.length > 0;
            const hasLoans = incomingLoans.length > 0;
            const hasReport = !!report?.summary_kpis;
            const hasAnalytics = analytics && Object.keys(analytics || {}).length > 0;
            const hasMetrics = metrics && Object.keys(metrics || {}).length > 0;
            const hasTimeline = !!(timeline?.dates?.length);
            const hasSamples = !!(samples?.investors?.length || samples?.loans?.length);

            // Capture baselines when we have real data
            if (hasInvestors) this.baselineInvestors = incomingInvestors;
            if (hasLoans) this.baselineLoans = incomingLoans;
            if (hasReport) this.baselineReport = report;
            if (hasAnalytics) this.baselineAnalytics = analytics;
            if (hasMetrics) this.baselineMetrics = metrics;
            if (hasTimeline) this.baselineTimeline = timeline;
            if (hasSamples) this.baselineSamples = samples;

            this.investors = hasInvestors ? incomingInvestors : (this.investors.length ? this.investors : this.baselineInvestors);
            this.loans = hasLoans ? incomingLoans : (this.loans.length ? this.loans : this.baselineLoans);
            this.report = hasReport ? report : (this.report || this.baselineReport);
            this.analyticsRaw = hasAnalytics ? analytics : (this.analyticsRaw || this.baselineAnalytics);
            this.modelMetrics = hasMetrics ? metrics : (this.modelMetrics || this.baselineMetrics);
            this.timeline = hasTimeline ? timeline : (this.timeline || this.baselineTimeline);
            if (hasSamples || (!this.samples && this.baselineSamples)) {
              this.samples = hasSamples ? samples : this.baselineSamples;
            }

            const metricsSrc = this.modelMetrics || {};
            const analyticsSrc = this.analyticsRaw || {};

            this.segmentHeatmaps = metricsSrc.segment_heatmaps || {};
            this.brokerHeatmaps = {
              churn: metricsSrc.segment_heatmaps?.churn_broker_by_tolerance,
              default: metricsSrc.segment_heatmaps?.default_broker_by_sector,
            };
            this.driftSummary = drift || this.driftSummary;
            this.dataAudit = audit || this.dataAudit;
            this.alerts = alerts || this.alerts;
            this.interventions = interventions || this.interventions;
            this.edaSummary = eda || this.edaSummary;
            this.aiKpiSnapshot = this.buildKpiSnapshot();
            this.predictorCharts = this.buildPredictorCharts();
            this.singleFeatureCharts = this.buildSingleFeatureCharts();
            this.diagnosticsCurves = this.buildDiagnosticsCharts();
            this.selectionCurves = this.buildSelectionCurves("churn", "rf_estimators");
            this.selectionCurvesDefault = this.buildSelectionCurves("default", "rf_estimators");
            this.selectionCurvesGb = this.buildSelectionCurves("churn", "gb_learning_rate");
            this.selectionCurvesGbDefault = this.buildSelectionCurves("default", "gb_learning_rate");
            this.nonLinearCharts = this.buildNonLinearCharts();
            this.imbalance = metricsSrc.imbalance || this.imbalance;
            this.imbalanceDetail = metricsSrc.imbalance || this.imbalanceDetail;
            this.robustness = metricsSrc.robustness || this.robustness;
            this.coeffStability = metricsSrc.bootstrap_coefficients || this.coeffStability || {};
            this.regressionKpi = metricsSrc.regression_kpi || this.regressionKpi;
            this.rocExplorer = this.buildRocExplorer();
            this.edaCharts = this.buildEdaCharts();
            this.fairnessCharts = this.buildFairnessCharts();
            this.forecaster = metricsSrc.forecaster || this.forecaster;
            this.missingHeatmap = audit?.heatmap || this.missingHeatmap;
            this.segmentBias = metricsSrc.segment_bias || this.segmentBias;
            this.segmentRocCharts = this.buildSegmentRocCharts();
            this.biasGapCharts = this.buildBiasCharts();
            this.votingCharts = this.buildVotingCharts();
            this.surfaceSlices = this.buildSurfaceSlices();

            const investorSource = hasInvestors ? incomingInvestors : this.investorRows;
            const loanSource = hasLoans ? incomingLoans : this.loanRows;
            const investorResolved = investorSource.length ? investorSource : (this.baselineInvestors || []);
            const loanResolved = loanSource.length ? loanSource : (this.baselineLoans || []);
            this.investorRows = investorResolved;
            this.loanRows = loanResolved;
            this.investorCount = this.investorRows.length;
            this.loanCount = this.loanRows.length;
            this.investorPage = 0;
            this.loanPage = 0;

            if (!this.selectedInvestorId && this.investorRows.length) {
              this.selectedInvestorId = this.investorRows[0]?.id ?? this.selectedInvestorId;
            }
            if (!this.selectedLoanId && this.loanRows.length) {
              this.selectedLoanId = this.loanRows[0]?.id ?? this.selectedLoanId;
            }

            this.dashboardContrib = this.buildDashboardContrib();
            this.updateCurveData();
            this.updateThresholdSummary();
            this.populateCharts(analyticsSrc);
            this.loading = false;
            this.cdr.detectChanges();
          },
          error: (err) => this.handleError("Failed to load dashboard data", err)
        });
      },
      error: (err) => this.handleError("Failed to refresh scores", err)
    });
  }

  refreshReport(): void {
    this.loading = true;
    this.api.generateReport().subscribe({
      next: (report) => {
        this.report = report;
        this.loading = false;
        this.snackBar.open("Report refreshed", "Close", { duration: 2000 });
      },
      error: (err) => this.handleError("Failed to refresh report", err)
    });
  }

  changeHorizon(h: "3m" | "6m" | "12m") {
    this.selectedHorizon = h;
    if (this.analyticsRaw) {
      this.populateCharts(this.analyticsRaw);
    }
    this.updateCurveData();
    this.updateThresholdSummary();
  }

  changeFamily(fam: string) {
    this.selectedFamily = fam as any;
    this.updateCurveData();
    this.updateRadar();
    this.updateComparisonCurves();
    this.updateThresholdSummary();
  }

  setSelectedInvestor(id: number) {
    this.selectedInvestorId = id;
    this.dashboardContrib = this.buildDashboardContrib();
  }

  setSelectedLoan(id: number) {
    this.selectedLoanId = id;
    this.dashboardContrib = this.buildDashboardContrib();
  }

  explainDashboard() {
    this.aiQuestion = "Explain today’s dashboard risk picture in 3 bullets.";
    this.askAi();
    this.scrollToExplain();
  }

  heatmap(name: string) {
    return this.segmentHeatmaps?.[name];
  }

  investorBuckets() {
    return this.report?.summary_kpis.investor_buckets || { Low: 0, Medium: 0, High: 0 };
  }

  loanBuckets() {
    return this.report?.summary_kpis.loan_buckets || { Low: 0, Medium: 0, High: 0 };
  }

  topInvestors() {
    return this.report?.summary_kpis.top_investors || [];
  }

  topLoans() {
    return this.report?.summary_kpis.top_loans || [];
  }

  formatPercent(prob?: number): string {
    if (prob === undefined || prob === null) {
      return "0.0%";
    }
    return `${(prob * 100).toFixed(1)}%`;
  }

  private populateCharts(analytics: any): void {
    if (!analytics) {
      this.charts.churnBuckets = { labels: ["Low", "Medium", "High"], data: [0, 0, 0] };
      this.charts.defaultBuckets = { labels: ["Low", "Medium", "High"], data: [0, 0, 0] };
      this.charts.sectorExposure = { labels: [], data: [] };
      this.charts.aumBuckets = { labels: [], data: [] };
      this.charts.yieldEngagementTrend = { labels: [], data: [] };
      this.charts.dscrSummary = { labels: ["p25", "p50", "p75"], data: [0, 0, 0] };
      return;
    }
    const h = this.selectedHorizon;
    const churnSrc = analytics.churn_buckets_by_horizon?.[h] || analytics.churn_buckets || {};
    const defaultSrc = analytics.default_buckets_by_horizon?.[h] || analytics.default_buckets || {};
    this.charts.churnBuckets = {
      labels: Object.keys(churnSrc || {}),
      data: Object.values(churnSrc || {})
    };
    this.charts.defaultBuckets = {
      labels: Object.keys(defaultSrc || {}),
      data: Object.values(defaultSrc || {})
    };
    const sortedSector = Object.entries(analytics.sector_exposure || {}).sort((a: any, b: any) => b[1].exposure - a[1].exposure);
    this.charts.sectorExposure = {
      labels: sortedSector.map((s: any) => s[0]),
      data: sortedSector.map((s: any) => s[1].exposure)
    };
    this.charts.aumBuckets = {
      labels: Object.keys(analytics.aum_buckets || {}),
      data: Object.values(analytics.aum_buckets || {})
    };
    // Create a simple line series for yield and engagement medians to show trend-like context
    const yieldMed = analytics.distribution_yield?.p50 ?? 0;
    const engMed = analytics.engagement?.p50 ?? 0;
    this.charts.yieldEngagementTrend = {
      labels: ["Engagement median", "Yield median"],
      data: [engMed, yieldMed * 100]
    };
    const dscr = analytics.dscr || {};
    this.charts.dscrSummary = {
      labels: ["p25", "p50", "p75"],
      data: [dscr.p25 ?? 0, dscr.p50 ?? 0, dscr.p75 ?? 0]
    };
    this.brokerPage = 0;
  }

  applyInvestorFilter(filterValue: string) {
    this.investorFilter = filterValue.trim().toLowerCase();
    this.investorPage = 0;
  }

  applyLoanFilter(filterValue: string) {
    this.loanFilter = filterValue.trim().toLowerCase();
    this.loanPage = 0;
  }

  filteredInvestors() {
    const term = this.investorFilter || "";
    if (!term) return this.investorRows;
    return this.investorRows.filter((i) =>
      [i.name, i.risk_tolerance, i.risk_bucket].some((v) => (v || "").toString().toLowerCase().includes(term))
    );
  }

  filteredLoans() {
    const term = this.loanFilter || "";
    if (!term) return this.loanRows;
    return this.loanRows.filter((l) =>
      [l.id, l.sector, l.risk_bucket].some((v) => (v || "").toString().toLowerCase().includes(term))
    );
  }

  private sortArray<T>(arr: T[], key: keyof T | "", dir: "asc" | "desc") {
    if (!key) return arr;
    return [...arr].sort((a: any, b: any) => {
      const va = a?.[key] ?? "";
      const vb = b?.[key] ?? "";
      if (va < vb) return dir === "asc" ? -1 : 1;
      if (va > vb) return dir === "asc" ? 1 : -1;
      return 0;
    });
  }

  sortedInvestors() {
    return this.sortArray(this.filteredInvestors(), this.investorSortKey, this.investorSortDir);
  }

  sortedLoans() {
    return this.sortArray(this.filteredLoans(), this.loanSortKey, this.loanSortDir);
  }

  pagedInvestors() {
    const start = this.investorPage * this.pageSize;
    return this.sortedInvestors().slice(start, start + this.pageSize);
  }

  pagedLoans() {
    const start = this.loanPage * this.pageSize;
    return this.sortedLoans().slice(start, start + this.pageSize);
  }

  setInvestorPage(delta: number) {
    const total = this.filteredInvestors().length;
    const maxPage = Math.max(0, Math.ceil(total / this.pageSize) - 1);
    this.investorPage = Math.min(maxPage, Math.max(0, this.investorPage + delta));
  }

  setLoanPage(delta: number) {
    const total = this.filteredLoans().length;
    const maxPage = Math.max(0, Math.ceil(total / this.pageSize) - 1);
    this.loanPage = Math.min(maxPage, Math.max(0, this.loanPage + delta));
  }

  changePageSize(size: number) {
    this.pageSize = size;
    this.investorPage = 0;
    this.loanPage = 0;
  }

  brokerRowsPaged() {
    const rows = this.brokerHeatmaps?.default?.rows || [];
    const start = this.brokerPage * this.brokerPageSize;
    return rows.slice(start, start + this.brokerPageSize);
  }

  brokerRowCount() {
    return this.brokerHeatmaps?.default?.rows?.length || 0;
  }

  setBrokerPage(delta: number) {
    const total = this.brokerRowCount();
    const maxPage = Math.max(0, Math.ceil(total / this.brokerPageSize) - 1);
    this.brokerPage = Math.min(maxPage, Math.max(0, this.brokerPage + delta));
  }

  toggleInvestorSort(key: keyof Investor) {
    if (this.investorSortKey === key) {
      this.investorSortDir = this.investorSortDir === "asc" ? "desc" : "asc";
    } else {
      this.investorSortKey = key;
      this.investorSortDir = "asc";
    }
    this.investorPage = 0;
  }

  toggleLoanSort(key: keyof Loan) {
    if (this.loanSortKey === key) {
      this.loanSortDir = this.loanSortDir === "asc" ? "desc" : "asc";
    } else {
      this.loanSortKey = key;
      this.loanSortDir = "asc";
    }
    this.loanPage = 0;
  }

  askAi(): void {
    if (!this.aiQuestion) return;
    this.aiLoading = true;
    this.aiHistory.push({ role: "user", content: this.aiQuestion });
    const context = {
      report: this.report,
      charts: this.charts,
      timeline: this.timeline,
      topInvestors: this.topInvestors(),
      topLoans: this.topLoans(),
      samples: this.samples,
      kpis: this.aiKpiSnapshot,
      metrics: this.modelMetrics,
    };
    this.api.askAi(this.aiQuestion, "dashboard", context, this.aiHistory).subscribe({
      next: (resp) => {
        this.aiAnswer = resp.answer;
        this.aiHistory.push({ role: "assistant", content: resp.answer });
        this.aiLoading = false;
        this.aiQuestion = "";
      },
      error: (err) => this.handleError("AI explanation failed", err)
    });
  }

  private buildKpiSnapshot() {
    const analytics = this.analyticsRaw || {};
    return {
      churn_buckets: analytics.churn_buckets_by_horizon?.[this.selectedHorizon] || analytics.churn_buckets,
      default_buckets: analytics.default_buckets_by_horizon?.[this.selectedHorizon] || analytics.default_buckets,
      sector_exposure: analytics.sector_exposure,
      cohort_risk: analytics.cohort_risk,
      deltas: this.report?.summary_kpis?.deltas,
      model_family_metrics: this.familyMetrics(),
    };
  }

  private handleError(message: string, err: any): void {
    console.error(message, err);
    this.loading = false;
    this.snackBar.open(message, "Close", { duration: 3000 });
  }

  familyMetrics() {
    const famBlock = this.modelMetrics?.model_families || {};
    const h = this.selectedHorizon;
    return {
      churn: famBlock.churn?.[this.selectedFamily]?.[h],
      default: famBlock.default?.[this.selectedFamily]?.[h]
    };
  }

  private thresholdsFor(problem: "churn" | "default") {
    const famBlock = this.modelMetrics?.model_families?.[problem]?.[this.selectedFamily]?.[this.selectedHorizon];
    const familyThresholds = famBlock?.thresholds || [];
    if (familyThresholds.length) return familyThresholds;
    return this.modelMetrics?.thresholds?.[problem]?.[this.selectedHorizon] || [];
  }

  cohortRisk() {
    return this.analyticsRaw?.cohort_risk || {};
  }

  private nearestThreshold(problem: "churn" | "default") {
    const thresholds = this.thresholdsFor(problem);
    if (!thresholds.length) return null;
    let best = thresholds[0];
    let bestDiff = Math.abs(best.threshold - this.thresholdValue);
    for (const t of thresholds) {
      const diff = Math.abs(t.threshold - this.thresholdValue);
      if (diff < bestDiff) {
        best = t;
        bestDiff = diff;
      }
    }
    return best;
  }

  thresholdStats() {
    return this.thresholdSummary;
  }

  onThresholdInput(val: number | { value: number | null } | null) {
    const next = typeof val === "number" ? val : val?.value ?? null;
    if (next === null || next === undefined) return;
    this.thresholdValue = next;
    this.updateThresholdSummary();
  }

  private updateThresholdSummary() {
    this.thresholdSummary = {
      churn: this.nearestThreshold("churn"),
      default: this.nearestThreshold("default"),
    };
  }

  segmentScores() {
    return this.analyticsRaw?.segment_scores || {};
  }

  predictorBenchmarks() {
    const bench = this.modelMetrics?.single_feature_benchmarks || {};
    const toRows = (obj: any) =>
      Object.entries(obj || {})
        .map(([name, val]: any) => ({ name, auc: val?.roc_auc || 0 }))
        .sort((a, b) => b.auc - a.auc);
    return {
      churn: toRows(bench.churn),
      default: toRows(bench.default),
    };
  }

  private buildPredictorCharts() {
    const bench = this.predictorBenchmarks();
    const churnLabels = bench.churn.map((r) => r.name);
    const churnData = bench.churn.map((r) => r.auc);
    const defLabels = bench.default.map((r) => r.name);
    const defData = bench.default.map((r) => r.auc);
    const familyAuc = this.modelMetrics?.model_families?.churn ? Object.entries(this.modelMetrics.model_families.churn).map(([fam, hv]: any) => ({
      family: fam,
      auc: hv?.["6m"]?.roc_auc || 0
    })) : [];
    const familyAucDefault = this.modelMetrics?.model_families?.default ? Object.entries(this.modelMetrics.model_families.default).map(([fam, hv]: any) => ({
      family: fam,
      auc: hv?.["6m"]?.roc_auc || 0
    })) : [];
    return {
      churn: {
        labels: churnLabels,
        datasets: [{ label: "Churn single-feature AUC", data: churnData, backgroundColor: "#0ea5e9" }],
      },
      default: {
        labels: defLabels,
        datasets: [{ label: "Default single-feature AUC", data: defData, backgroundColor: "#f59e0b" }],
      },
      families: {
        labels: familyAuc.map((r: any) => r.family),
        datasets: [{ label: "Churn AUC by family (6m)", data: familyAuc.map((r: any) => r.auc), backgroundColor: "#10b981" }],
      },
      familiesDefault: {
        labels: familyAucDefault.map((r: any) => r.family),
        datasets: [{ label: "Default AUC by family (6m)", data: familyAucDefault.map((r: any) => r.auc), backgroundColor: "#6366f1" }],
      }
    };
  }

  private buildDiagnosticsCharts() {
    const diag = this.modelMetrics?.diagnostics_1d || {};
    const makeLine = (d: any, colorA: string, colorB: string, labelA: string, labelB: string) => {
      if (!d?.x) return undefined;
      return {
        labels: d.x,
        datasets: [
          { data: d.logistic || [], label: labelA, borderColor: colorA, backgroundColor: "rgba(0,0,0,0)", tension: 0.2, fill: false },
          { data: d.knn || [], label: labelB, borderColor: colorB, backgroundColor: "rgba(0,0,0,0)", tension: 0.2, fill: false },
        ],
      };
    };
    return {
      churn: makeLine(diag.churn, "#22c55e", "#ef4444", "Logistic", "k-NN"),
      default: makeLine(diag.default, "#3b82f6", "#f97316", "Logistic", "k-NN"),
    };
  }

  private buildSingleFeatureCharts() {
    const sf = this.modelMetrics?.single_feature_benchmarks || {};
    const make = (obj: any, color: string, label: string) => {
      if (!obj) return undefined;
      const entries = Object.entries(obj);
      return {
        labels: entries.map((e: any) => e[0]),
        datasets: [{ label, data: entries.map((e: any) => (e[1]?.roc_auc || 0)), backgroundColor: color }],
      };
    };
    return {
      churn: make(sf.churn, "#22c55e", "Single feature AUC"),
      default: make(sf.default, "#ef4444", "Single feature AUC"),
    };
  }

  private buildSegmentRocCharts() {
    const bias = this.modelMetrics?.segment_bias || {};
    const make = (rows: any[], color: string, label: string) => {
      if (!rows?.length) return undefined;
      return {
        labels: rows.map((r) => r.segment),
        datasets: [{ label, data: rows.map((r) => r.auc || 0), backgroundColor: color }],
      };
    };
    return {
      churn: make(bias.churn?.by_segment, "#10b981", "AUC by tolerance"),
      default: make(bias.default?.by_segment, "#6366f1", "AUC by sector"),
    };
  }

  private buildBiasCharts() {
    const bias = this.modelMetrics?.segment_bias || {};
    const make = (rows: any[], label: string, key: "tpr_gap" | "fpr_gap", color: string) => {
      if (!rows?.length) return undefined;
      return {
        labels: rows.map((r) => r.segment),
        datasets: [{ label, data: rows.map((r) => (r?.[key] || 0) * 100), backgroundColor: color }],
      };
    };
    return {
      churn: make(bias.churn?.by_segment, "TPR gap vs overall (%)", "tpr_gap", "#f59e0b"),
      default: make(bias.default?.by_segment, "TPR gap vs overall (%)", "tpr_gap", "#f97316"),
    };
  }

  private buildVotingCharts() {
    const voting = this.modelMetrics?.voting_importance || {};
    const make = (rows: any[], color: string) => {
      if (!rows?.length) return undefined;
      const top = rows.slice(0, 8);
      return {
        labels: top.map((r) => r.feature),
        datasets: [{ label: "Voting score", data: top.map((r) => r.score || 0), backgroundColor: color }],
      };
    };
    return {
      churn: make(voting.churn, "#0ea5e9"),
      default: make(voting.default, "#a855f7"),
    };
  }

  private buildSurfaceSlices() {
    const surf = this.modelMetrics?.surfaces || {};
    const build = (obj: any, color: string) => {
      if (!obj?.x || !obj?.y || !obj?.z) return undefined;
      const datasets = (obj.y as number[]).slice(0, 3).map((yVal: number, idx: number) => ({
        label: `${obj.y_label || "y"}=${yVal.toFixed(1)}`,
        data: obj.z[idx] as number[],
        borderColor: color,
        backgroundColor: "rgba(0,0,0,0)",
        tension: 0.15,
        fill: false,
      }));
      return { labels: obj.x as number[], datasets };
    };
    return {
      churn: build(surf.churn, "#22c55e"),
      default: build(surf.default, "#ef4444"),
    };
  }

  private buildDashboardContrib() {
    const contrib = this.modelMetrics?.contributions || {};
    const investor = this.selectedInvestorId ? contrib.churn?.[this.selectedInvestorId] : null;
    const loan = this.selectedLoanId ? contrib.default?.[this.selectedLoanId] : null;
    const make = (obj: any, color: string, label: string) =>
      obj
        ? {
            labels: Object.keys(obj),
            datasets: [{ label, data: Object.values(obj).map((v: any) => Number(v) || 0), backgroundColor: color }],
          }
        : undefined;
    return {
      investor: make(investor, "#0ea5e9", "Contribution"),
      loan: make(loan, "#f97316", "Contribution"),
    };
  }

  private buildSelectionCurves(problem: "churn" | "default", key: "rf_estimators" | "gb_learning_rate") {
    const sel = this.modelMetrics?.model_selection?.[problem]?.[key];
    if (!sel) return { datasets: [] };
    const params = sel.params || [];
    const toPoints = (arr: number[]) => (arr || []).map((y: number, idx: number) => ({ x: params[idx] ?? idx, y }));
    return {
      datasets: [
        { label: "Train AUC", data: toPoints(sel.train || []), borderColor: "#22c55e", tension: 0.15, fill: false },
        { label: "Val AUC", data: toPoints(sel.val || []), borderColor: "#f59e0b", tension: 0.15, fill: false },
        { label: "CV AUC", data: toPoints(sel.cv || []), borderColor: "#6366f1", tension: 0.15, fill: false },
      ],
    };
  }

  selectionSeedTable() {
    const sel = this.modelMetrics?.model_selection?.churn?.rf_estimators;
    const seeds = this.modelMetrics?.model_selection?.churn?.seeds || [1, 7, 21];
    if (!sel?.seed_runs) return { seeds: [], rows: [] };
    const rows = (sel.seed_runs as number[][]).map((run: number[], idx: number) => ({
      param: sel.params?.[idx],
      scores: run || [],
      mean: run && run.length ? run.reduce((a, b) => a + b, 0) / run.length : 0,
    }));
    return { seeds, rows };
  }

  missingnessGrid(kind: "investor" | "loan") {
    const hm = this.missingHeatmap?.[kind];
    if (!hm?.rows || !hm?.cols) return null;
    return hm;
  }

  private buildRocExplorer() {
    const thresholds = this.thresholdsFor("churn");
    const points = thresholds.map((t: any) => ({ x: t.fpr, y: t.tpr }));
    const scenarioPts = [0.33, 0.5, 0.66].map((thr) => {
      const nearest = thresholds.reduce((prev: any, curr: any) =>
        Math.abs(curr.threshold - thr) < Math.abs((prev?.threshold ?? 0) - thr) ? curr : prev
      , thresholds[0] || {});
      return { x: nearest?.fpr || 0, y: nearest?.tpr || 0 };
    });
    return {
      datasets: [
        { label: "Churn ROC", data: points, borderColor: "#0ea5e9", backgroundColor: "rgba(14,165,233,0.2)", showLine: true, tension: 0.1, fill: false },
        { label: "Scenario thresholds", data: scenarioPts, borderColor: "#f59e0b", backgroundColor: "rgba(245,158,11,0.4)", showLine: false, pointRadius: 5, fill: false },
      ]
    };
  }

  private buildNonLinearCharts() {
    const nl = this.modelMetrics?.non_linear || {};
    const make = (series: any, label: string, color: string) => {
      if (!series) return undefined;
      return {
        labels: series.x,
        datasets: [{ data: series.y, label, borderColor: color, tension: 0.15, fill: false }],
      };
    };
    return {
      churn: make(nl.churn?.engagement_score, "Churn vs engagement", "#22c55e"),
      default: make(nl.default?.ltv_ratio, "Default vs LTV", "#ef4444"),
    };
  }

  private buildEdaCharts() {
    const eda = this.modelMetrics?.eda;
    if (!eda) return {};
    const invHist = eda.investor?.hist || {};
    const loanHist = eda.loan?.hist || {};
    const makeHist = (hist: any) => ({
      labels: (hist.bins || []).slice(0, -1).map((b: number, idx: number) => `${b.toFixed(1)}-${hist.bins[idx+1]?.toFixed(1)}`),
      datasets: [{ data: hist.counts || [], backgroundColor: "#60a5fa" }],
    });
    const invCorr = eda.investor?.corr || {};
    const loanCorr = eda.loan?.corr || {};
    return {
      investor: {
        engagement: invHist.engagement_score ? makeHist(invHist.engagement_score) : null,
        email_open: invHist.email_open_rate ? makeHist(invHist.email_open_rate) : null,
        age: invHist.age ? makeHist(invHist.age) : null,
        corr: invCorr,
        outliers: eda.investor?.outliers,
      },
      loan: {
        ltv: loanHist.ltv_ratio ? makeHist(loanHist.ltv_ratio) : null,
        dscr: loanHist.dscr ? makeHist(loanHist.dscr) : null,
        corr: loanCorr,
        outliers: eda.loan?.outliers,
      },
    };
  }

  private buildFairnessCharts() {
    const seg = this.modelMetrics?.segment_roc || {};
    const toBar = (obj: any, color: string, label: string) =>
      obj ? {
        labels: Object.keys(obj || {}),
        datasets: [{ label, data: Object.values(obj || {}) as number[], backgroundColor: color }],
      } : undefined;
    const churn = toBar(seg.churn, "#22c55e", "Churn AUC");
    const churnAp = toBar(seg.churn_ap, "#0ea5e9", "Churn AP");
    const def = toBar(seg.default, "#f59e0b", "Default AUC");
    const defAp = toBar(seg.default_ap, "#8b5cf6", "Default AP");
    return { churn, churnAp, default: def, defAp };
  }

  hyperparams() {
    return this.modelMetrics?.hyperparameters || {};
  }

  sensitivityRows() {
    const sens = this.modelMetrics?.sensitivity || {};
    const toRows = (obj: any) =>
      Object.entries(obj || {})
        .map(([feat, vals]: any) => ({
          feature: feat,
          up: (vals?.up_flip_pct || 0) * 100,
          down: (vals?.down_flip_pct || 0) * 100
        }))
        .sort((a, b) => b.up - a.up);
    return { churn: toRows(sens.churn), default: toRows(sens.default) };
  }

  thresholdCost(problem: "churn" | "default") {
    const stats = this.thresholdSummary?.[problem];
    if (!stats) return null;
    const fp = stats.fp || 0;
    const fn = stats.fn || 0;
    return {
      fp,
      fn,
      cost_fp: fp * this.costFP,
      cost_fn: fn * this.costFN,
      total: fp * this.costFP + fn * this.costFN
    };
  }

  private portfolioScenarioSummary(resp: any) {
    const invBaseBuckets = resp?.investors?.base_buckets || resp?.base?.investor_buckets || {};
    const invScenarioBuckets = resp?.investors?.scenario_buckets || resp?.scenario?.investor_buckets || {};
    const loanBaseBuckets = resp?.loans?.base_buckets || resp?.base?.loan_buckets || {};
    const loanScenarioBuckets = resp?.loans?.scenario_buckets || resp?.scenario?.loan_buckets || {};
    const highCount = (obj: any) => Number(obj?.High ?? obj?.high ?? 0);
    return {
      horizon: resp?.horizon || this.selectedHorizon,
      base: {
        avg_churn: resp?.base?.avg_churn ?? resp?.investors?.base_avg ?? null,
        avg_default: resp?.base?.avg_default ?? resp?.loans?.base_avg ?? null,
        high_risk_investors: highCount(invBaseBuckets),
        high_risk_loans: highCount(loanBaseBuckets),
      },
      scenario: {
        avg_churn: resp?.scenario?.avg_churn ?? resp?.investors?.scenario_avg ?? null,
        avg_default: resp?.scenario?.avg_default ?? resp?.loans?.scenario_avg ?? null,
        high_risk_investors: highCount(invScenarioBuckets),
        high_risk_loans: highCount(loanScenarioBuckets),
      },
      buckets: {
        investors: { base: invBaseBuckets, scenario: invScenarioBuckets },
        loans: { base: loanBaseBuckets, scenario: loanScenarioBuckets },
      }
    };
  }

  runPortfolioScenario() {
    this.loading = true;
    this.api.runPortfolioScenario({ horizon: this.selectedHorizon }).subscribe({
      next: (resp) => {
        this.scenarioResult = this.portfolioScenarioSummary(resp);
        // Re-load dashboard data so tables/charts aren't left empty after scenario adjustments
        this.loadData();
        this.snackBar.open("Scenario calculated", "Close", { duration: 2000 });
      },
      error: (err) => this.handleError("Failed to run portfolio scenario", err)
    });
  }

  imputationRows() {
    const missing = this.modelMetrics?.missing_data || {};
    const toRows = (obj: any) =>
      Object.entries(obj || {}).map(([name, vals]: any) => ({
        strategy: name,
        auc: vals?.roc_auc || 0,
        ap: vals?.avg_precision || 0,
        retained: (vals?.retained_pct || 0) * 100,
      }));
    const churnRows = toRows(missing.churn);
    const defaultRows = toRows(missing.default);
    const bestChurn = [...churnRows].sort((a, b) => b.auc - a.auc)[0];
    const bestDefault = [...defaultRows].sort((a, b) => b.auc - a.auc)[0];
    return { churn: churnRows, default: defaultRows, bestChurn, bestDefault };
  }

  bootstrapStats() {
    return this.modelMetrics?.bootstrap_ci || {};
  }

  cohortModels() {
    return this.modelMetrics?.cohort_models || {};
  }

  selectInvestor(inv: Investor) {
    this.selectedInvestor = inv;
    this.selectedLoan = null;
  }

  selectLoan(loan: Loan) {
    this.selectedLoan = loan;
    this.selectedInvestor = null;
  }

  private scrollToExplain() {
    if (!this.explainAnchor) {
      this.explainAnchor = document.getElementById("ai-explain-card");
    }
    const target = this.explainAnchor;
    if (target?.scrollIntoView) {
      target.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  explainCard(id: string, prompt: string, context: any = {}): void {
    const existing = this.cardExplain[id] || { text: "", loading: false };
    this.cardExplain[id] = { ...existing, loading: true };
    this.api.askAi(prompt, "dashboard-card", context, []).subscribe({
      next: (resp) => (this.cardExplain[id] = { text: resp.answer, loading: false }),
      error: () => {
        this.cardExplain[id] = { text: "Unable to fetch explanation right now.", loading: false };
        this.snackBar.open("Explain failed", "Close", { duration: 2000 });
      }
    });
  }

  private updateCurveData() {
    const churnThr = this.thresholdsFor("churn");
    const defaultThr = this.thresholdsFor("default");
    this.rocData = {
      datasets: [
        {
          label: "Churn ROC",
          data: churnThr.map((t: any) => ({ x: t.fpr, y: t.tpr })),
          borderColor: "#19c3b1",
          backgroundColor: "rgba(25,195,177,0.2)",
          tension: 0.15,
          showLine: true,
          fill: false
        },
        {
          label: "Default ROC",
          data: defaultThr.map((t: any) => ({ x: t.fpr, y: t.tpr })),
          borderColor: "#4f46e5",
          backgroundColor: "rgba(79,70,229,0.2)",
          tension: 0.15,
          showLine: true,
          fill: false
        }
      ]
    };
    this.prData = {
      datasets: [
        {
          label: "Churn PR",
          data: churnThr.map((t: any) => ({ x: t.recall, y: t.precision })),
          borderColor: "#10b981",
          backgroundColor: "rgba(16,185,129,0.2)",
          tension: 0.15,
          showLine: true,
          fill: false
        },
        {
          label: "Default PR",
          data: defaultThr.map((t: any) => ({ x: t.recall, y: t.precision })),
          borderColor: "#f59e0b",
          backgroundColor: "rgba(245,158,11,0.2)",
          tension: 0.15,
          showLine: true,
          fill: false
        }
      ]
    };
    this.updateRadar();
    this.updateComparisonCurves();
  }

  private updateRadar() {
    const fam = this.familyMetrics();
    const churn = fam?.churn || {};
    const defm = fam?.default || {};
    const labels = ["Accuracy", "Precision", "Recall", "ROC AUC", "Avg Precision"];
    const churnVals = [churn.accuracy || 0, churn.precision || 0, churn.recall || 0, churn.roc_auc || 0, churn.avg_precision || 0];
    const defVals = [defm.accuracy || 0, defm.precision || 0, defm.recall || 0, defm.roc_auc || 0, defm.avg_precision || 0];
    this.radarData = {
      labels,
      datasets: [
        { label: "Churn", data: churnVals, borderColor: "#19c3b1", backgroundColor: "rgba(25,195,177,0.2)" },
        { label: "Default", data: defVals, borderColor: "#4f46e5", backgroundColor: "rgba(79,70,229,0.2)" }
      ]
    };
    const churnBuckets = [churn.bucket_low || 0, churn.bucket_med || 0, churn.bucket_high || 0];
    const defaultBuckets = [defm.bucket_low || 0, defm.bucket_med || 0, defm.bucket_high || 0];
    this.bucketData = {
      labels: ["Low", "Medium", "High"],
      datasets: [
        { label: "Churn buckets", data: churnBuckets, backgroundColor: "rgba(25,195,177,0.4)" },
        { label: "Default buckets", data: defaultBuckets, backgroundColor: "rgba(79,70,229,0.4)" },
      ]
    };
  }

  changeCompareFamily(f: string) {
    this.compareFamily = f as any;
    this.updateComparisonCurves();
  }

  private thresholdsForFamily(problem: "churn" | "default", family: string) {
    const famBlock = this.modelMetrics?.model_families?.[problem]?.[family]?.[this.selectedHorizon];
    return famBlock?.thresholds || [];
  }

  private updateComparisonCurves() {
    const famA = this.selectedFamily;
    const famB = this.compareFamily;
    const churnA = this.thresholdsForFamily("churn", famA);
    const churnB = this.thresholdsForFamily("churn", famB);
    const defA = this.thresholdsForFamily("default", famA);
    const defB = this.thresholdsForFamily("default", famB);
    this.compareRoc = {
      datasets: [
        { label: `Churn ${famA}`, data: churnA.map((t: any) => ({ x: t.fpr, y: t.tpr })), borderColor: "#19c3b1", fill: false, tension: 0.1 },
        { label: `Churn ${famB}`, data: churnB.map((t: any) => ({ x: t.fpr, y: t.tpr })), borderColor: "#f97316", fill: false, tension: 0.1 },
        { label: `Default ${famA}`, data: defA.map((t: any) => ({ x: t.fpr, y: t.tpr })), borderColor: "#4f46e5", fill: false, tension: 0.1 },
        { label: `Default ${famB}`, data: defB.map((t: any) => ({ x: t.fpr, y: t.tpr })), borderColor: "#22c55e", fill: false, tension: 0.1 },
      ]
    };
    this.comparePr = {
      datasets: [
        { label: `Churn ${famA}`, data: churnA.map((t: any) => ({ x: t.recall, y: t.precision })), borderColor: "#10b981", fill: false, tension: 0.1 },
        { label: `Churn ${famB}`, data: churnB.map((t: any) => ({ x: t.recall, y: t.precision })), borderColor: "#f59e0b", fill: false, tension: 0.1 },
        { label: `Default ${famA}`, data: defA.map((t: any) => ({ x: t.recall, y: t.precision })), borderColor: "#6366f1", fill: false, tension: 0.1 },
        { label: `Default ${famB}`, data: defB.map((t: any) => ({ x: t.recall, y: t.precision })), borderColor: "#ef4444", fill: false, tension: 0.1 },
      ]
    };
  }
}
