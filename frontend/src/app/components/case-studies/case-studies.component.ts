import { Component, OnInit } from "@angular/core";
import { MatSnackBar } from "@angular/material/snack-bar";

import { ApiService } from "../../services/api.service";
import { CaseStudy } from "../../models/case-study.model";

@Component({
  selector: "app-case-studies",
  templateUrl: "./case-studies.component.html",
  styleUrls: ["./case-studies.component.scss"],
})
export class CaseStudiesComponent implements OnInit {
  caseStudies: CaseStudy[] = [];
  loading = false;

  constructor(private api: ApiService, private snackBar: MatSnackBar) {}

  ngOnInit(): void {
    this.fetch();
  }

  fetch(): void {
    this.loading = true;
    this.api.getCaseStudies().subscribe({
      next: (data) => {
        this.caseStudies = data;
        this.loading = false;
      },
      error: (err) => {
        console.error(err);
        this.loading = false;
        this.snackBar.open("Failed to load case studies", "Close", { duration: 2500 });
      },
    });
  }
}
