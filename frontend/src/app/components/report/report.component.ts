import { Component, OnInit } from "@angular/core";
import { MatSnackBar } from "@angular/material/snack-bar";
import { ChartConfiguration } from "chart.js";

import { ApiService } from "../../services/api.service";
import { ReportResponse } from "../../models/report.model";

@Component({
  selector: "app-report",
  templateUrl: "./report.component.html",
  styleUrls: ["./report.component.scss"]
})
export class ReportComponent implements OnInit {
  report?: ReportResponse;
  loading = false;
  trendEngagement: ChartConfiguration<"line">["data"] = { labels: [], datasets: [] };
  trendDscr: ChartConfiguration<"line">["data"] = { labels: [], datasets: [] };
  deltas: any;
  playbooks: any;
  aiQuestion = "";
  aiAnswer = "";
  aiLoading = false;
  aiHistory: { role: string; content: string }[] = [];
  aiOpen = true;
  segmentType = "";
  segmentValue = "";
  reportFormat = "full";
  formats = [
    { value: "full", label: "Full report" },
    { value: "board", label: "Board summary" },
    { value: "rm", label: "RM call notes" },
    { value: "ic", label: "IC pack highlights" },
  ];

  constructor(private api: ApiService, private snackBar: MatSnackBar) {}

  ngOnInit(): void {
    this.loadReport();
  }

  loadReport(): void {
    this.loading = true;
    // Refresh scores then generate a report to reduce backend training edge-cases.
    this.api
      .batchRefresh()
      .subscribe({
        next: () => this.api.generateReport(this.segmentType ? { segment_type: this.segmentType, segment_value: this.segmentValue, format: this.reportFormat } : { format: this.reportFormat }).subscribe({
          next: (report) => {
            this.report = report;
            this.trendEngagement = this.toTrendData(report.summary_kpis.engagement_trend || [], "Engagement");
            this.trendDscr = this.toTrendData(report.summary_kpis.dscr_trend || [], "DSCR");
            this.deltas = report.summary_kpis.deltas;
            this.playbooks = report.summary_kpis.playbooks;
            this.loading = false;
          },
          error: (err) => this.handleError("Failed to load report", err)
        }),
        error: (err) => this.handleError("Failed to refresh scores", err)
      });
  }

  copyReport(): void {
    if (!this.report) return;
    navigator.clipboard
      .writeText(this.report.report_markdown)
      .then(() => this.snackBar.open("Report copied to clipboard", "Close", { duration: 2000 }))
      .catch(() => this.snackBar.open("Unable to copy", "Close", { duration: 2000 }));
  }

  printPdf(): void {
    this.api
      .fetchReportPdf(this.segmentType ? { segment_type: this.segmentType, segment_value: this.segmentValue, format: this.reportFormat } : { format: this.reportFormat })
      .subscribe({
        next: (html) => {
          const w = window.open("", "_blank");
          if (w) {
            w.document.write(html);
            w.document.close();
            w.focus();
            w.print();
          } else {
            this.snackBar.open("Pop-up blocked; allow pop-ups to print.", "Close", { duration: 3000 });
          }
        },
        error: (err) => this.handleError("Failed to open PDF view", err)
      });
  }

  downloadReport(): void {
    if (!this.report) return;
    const blob = new Blob([this.report.report_markdown], { type: "text/markdown;charset=utf-8" });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `capspace-report-${new Date().toISOString().slice(0,10)}.md`;
    a.click();
    window.URL.revokeObjectURL(url);
  }

  downloadPdf(): void {
    this.api.fetchReportPdf(this.segmentType ? { segment_type: this.segmentType, segment_value: this.segmentValue, format: this.reportFormat } : { format: this.reportFormat })
      .subscribe({
        next: (html) => {
          const blob = new Blob([html], { type: "text/html" });
          const url = URL.createObjectURL(blob);
          const w = window.open(url, "_blank");
          if (!w) {
            this.snackBar.open("Popup blocked. Please allow popups to view PDF.", "Close", { duration: 3000 });
          }
        },
        error: () => this.snackBar.open("Unable to generate PDF", "Close", { duration: 3000 })
      });
  }

  formatPercent(value?: number): string {
    if (value === undefined || value === null) return "0.0%";
    return `${(value * 100).toFixed(1)}%`;
  }

  sectorEntries(sectorDefault: Record<string, { high: number; exposure: number }>) {
    return Object.entries(sectorDefault || {});
  }

  // Expose calibration data for UI (used if we surface it)
  getCalibration(modelMetrics: any) {
    return modelMetrics?.calibration || {};
  }

  private toTrendData(series: number[], label: string) {
    const labels = series.map((_, idx) => `t${idx + 1}`);
    return {
      labels,
      datasets: [
        {
          data: series,
          label,
          borderColor: "#19c3b1",
          backgroundColor: "rgba(25,195,177,0.2)",
          tension: 0.25,
          fill: true,
        }
      ]
    };
  }

  askAi(): void {
    if (!this.aiQuestion) return;
    this.aiLoading = true;
    this.aiHistory.push({ role: "user", content: this.aiQuestion });
    const context = {
      report: this.report,
      trends: { engagement: this.trendEngagement, dscr: this.trendDscr },
      base_report_markdown: this.report?.base_report_markdown,
    };
    this.api.askAi(this.aiQuestion, "report", context, this.aiHistory).subscribe({
      next: (resp) => {
        this.aiAnswer = resp.answer;
        this.aiHistory.push({ role: "assistant", content: resp.answer });
        this.aiLoading = false;
        this.aiQuestion = "";
      },
      error: (err) => this.handleError("AI explanation failed", err)
    });
  }

  explainReportTemplate(template: string) {
    if (!this.report) return;
    this.aiQuestion = `Rewrite the daily report in style: ${template}`;
    this.askAi();
  }

  private handleError(message: string, err: any): void {
    console.error(message, err);
    this.loading = false;
    this.snackBar.open(message, "Close", { duration: 3000 });
  }
}
