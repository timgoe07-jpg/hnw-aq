import { Component, Input } from "@angular/core";
import { ProfileMatchResult } from "../../models/match-result.model";

@Component({
  selector: "app-match-results",
  templateUrl: "./match-results.component.html",
  styleUrls: ["./match-results.component.scss"],
})
export class MatchResultsComponent {
  @Input() results: ProfileMatchResult[] | null = null;
}
