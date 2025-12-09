import { Component } from "@angular/core";
import { MatSnackBar } from "@angular/material/snack-bar";
import { ApiService } from "../../services/api.service";
import { Profile } from "../../models/profile.model";
import { ProfileMatchResult } from "../../models/match-result.model";

@Component({
  selector: "app-profile-upload",
  templateUrl: "./profile-upload.component.html",
  styleUrls: ["./profile-upload.component.scss"],
})
export class ProfileUploadComponent {
  jsonInput = `[
  {
    "full_name": "Alex Smith",
    "current_title": "CIO",
    "current_company": "Aurora Capital",
    "industry": "Investment Management",
    "about_summary": "Allocates to private credit, prefers low vol strategies."
  }
]`;
  filterPersona = "";
  filterMinScore = 0;
  autoKeyword = "private credit infrastructure";
  autoLocation = "London";
  results: ProfileMatchResult[] = [];
  autoResults: ProfileMatchResult[] = [];
  loading = false;
  autoLoading = false;

  constructor(private api: ApiService, private snackBar: MatSnackBar) {}

  submitJson(): void {
    let profiles: Profile[] = [];
    try {
      profiles = JSON.parse(this.jsonInput);
      if (!Array.isArray(profiles)) throw new Error("JSON must be an array");
    } catch (e) {
      this.snackBar.open("Invalid JSON payload", "Close", { duration: 2500 });
      return;
    }
    this.loading = true;
    this.api.matchProfiles({ profiles }).subscribe({
      next: (resp) => {
        this.results = resp.results || resp.profiles || [];
        this.loading = false;
      },
      error: (err) => {
        console.error(err);
        this.loading = false;
        this.snackBar.open("Matching failed", "Close", { duration: 2500 });
      },
    });
  }

  runAutoSearch(): void {
    this.autoLoading = true;
    const payload = {
      keywords: this.autoKeyword,
      filters: {
        location: this.autoLocation || undefined,
      },
      max_results: 5,
    };
    this.api.getSuggestedProfiles(payload).subscribe({
      next: (resp) => {
        this.autoResults = resp.results || resp.profiles || [];
        this.autoLoading = false;
      },
      error: (err) => {
        console.error(err);
        this.autoLoading = false;
        this.snackBar.open("LinkedIn auto-match failed", "Close", { duration: 2500 });
      },
    });
  }
}
