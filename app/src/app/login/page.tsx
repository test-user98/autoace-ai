export const dynamic = "force-dynamic";

export default async function LoginPage({
  searchParams,
}: {
  // Next 15: searchParams is a Promise. Reading it here (rather than via
  // useSearchParams in a client component) avoids needing a Suspense boundary.
  searchParams: Promise<{ error?: string }>;
}) {
  const { error } = await searchParams;

  return (
    <main className="centered">
      <form className="card login" method="post" action="/api/login">
        <h1>AutoAce Voice Trial</h1>
        <p className="muted">Batch tone &amp; background-noise analysis.</p>

        {error ? <p className="error">Incorrect username or password.</p> : null}

        <label htmlFor="username">Username</label>
        <input id="username" name="username" autoComplete="username" required />

        <label htmlFor="password">Password</label>
        <input
          id="password"
          name="password"
          type="password"
          autoComplete="current-password"
          required
        />

        <button type="submit">Sign in</button>
      </form>
    </main>
  );
}
