import { Component } from "@angular/core";

interface ProjectionPoint {
  month: number;
  balance: number;
  interestEarned: number;
}

@Component({
  selector: "app-calculator",
  templateUrl: "./calculator.component.html",
  styleUrls: ["./calculator.component.scss"],
})
export class CalculatorComponent {
  investment = 1_000_000;
  annualRate = 10; // percent
  termMonths = 12;
  reinvest = true;
  monthlyPayout = false;

  projections: ProjectionPoint[] = [];
  summary = {
    monthlyIncome: 0,
    annualIncome: 0,
    totalInterest: 0,
    endingBalance: 0,
  };

  constructor() {
    this.calculate();
  }

  calculate(): void {
    const principal = Math.max(this.investment, 0);
    const r = this.annualRate / 100;
    const months = Math.max(this.termMonths, 1);
    const monthlyRate = r / 12;

    let balance = principal;
    let totalInterest = 0;
    const points: ProjectionPoint[] = [];

    for (let m = 1; m <= months; m++) {
      const interest = balance * monthlyRate;
      totalInterest += interest;
      if (this.reinvest) {
        balance += interest;
      }
      points.push({ month: m, balance, interestEarned: interest });
    }

    const monthlyIncome = principal * monthlyRate;
    const annualIncome = principal * r;

    this.projections = points;
    this.summary = {
      monthlyIncome: this.monthlyPayout ? monthlyIncome : points.at(-1)?.interestEarned || monthlyIncome,
      annualIncome,
      totalInterest,
      endingBalance: balance,
    };
  }

  get yearlyBalances(): { year: number; balance: number }[] {
    const perYear: { year: number; balance: number }[] = [];
    if (!this.projections.length) return perYear;
    const years = Math.ceil(this.termMonths / 12);
    for (let y = 1; y <= years; y++) {
      const month = Math.min(y * 12, this.termMonths);
      const point = this.projections.find((p) => p.month === month);
      if (point) perYear.push({ year: y, balance: point.balance });
    }
    return perYear;
  }
}
