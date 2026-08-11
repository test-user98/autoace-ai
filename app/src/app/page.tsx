import BatchRunner from "./BatchRunner";

export const dynamic = "force-dynamic";

export default function DashboardPage() {
  return (
    <main>
      <div className="topbar">
        <div>
          <h1>AutoAce Voice Trial</h1>
          <p className="muted">
            Upload a batch ZIP (audio files plus a <code>labels.csv</code> manifest). Analysis runs
            on Modal; this dashboard never processes audio.
          </p>
        </div>
        <form method="post" action="/api/logout">
          <button className="secondary" type="submit">
            Sign out
          </button>
        </form>
      </div>
      <BatchRunner />
    </main>
  );
}
