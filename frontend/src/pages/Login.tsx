import { ArrowRight, BarChart3, Camera, ShieldCheck } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/Button";
import { useAuthStore } from "@/stores/authStore";

export function Login() {
  const login = useAuthStore((state) => state.login);
  const isLoading = useAuthStore((state) => state.isLoading);
  const error = useAuthStore((state) => state.error);
  const [cpf, setCpf] = useState("");
  const [password, setPassword] = useState("");

  return (
    <main className="flex min-h-screen items-center justify-center bg-background p-4">
      <section className="grid w-full max-w-6xl overflow-hidden rounded-lg border border-border bg-card shadow-xl shadow-slate-200/70 lg:grid-cols-[1.08fr_0.92fr]">
        <div className="relative flex min-h-[560px] flex-col justify-between overflow-hidden bg-[url('/classroom-placeholder.svg')] bg-cover bg-center p-8 text-white">
          <div className="absolute inset-0 bg-slate-950/62" />
          <div className="relative flex h-11 w-11 items-center justify-center rounded-md bg-white/15 backdrop-blur">
            <ShieldCheck className="h-6 w-6" />
          </div>
          <div className="relative">
            <p className="text-sm font-medium uppercase tracking-wide text-white/75">SEDUC</p>
            <h1 className="mt-3 max-w-md text-4xl font-semibold leading-tight">Monitoramento comportamental em tempo real</h1>
            <div className="mt-6 grid max-w-lg gap-3 sm:grid-cols-2">
              <div className="rounded-lg border border-white/15 bg-white/10 p-4 backdrop-blur">
                <Camera className="h-5 w-5" />
                <p className="mt-3 text-sm font-medium">Video e status</p>
              </div>
              <div className="rounded-lg border border-white/15 bg-white/10 p-4 backdrop-blur">
                <BarChart3 className="h-5 w-5" />
                <p className="mt-3 text-sm font-medium">Indicadores pedagogicos</p>
              </div>
            </div>
          </div>
        </div>

        <form
          className="flex flex-col justify-center gap-5 p-8 lg:p-10"
          onSubmit={(event) => {
            event.preventDefault();
            void login({ cpf, password });
          }}
        >
          <div>
            <p className="text-sm font-medium text-primary">Bem-vindo</p>
            <h2 className="mt-1 text-2xl font-semibold tracking-tight">Acessar painel</h2>
            <p className="mt-2 text-sm text-muted-foreground">Entre com suas credenciais para continuar.</p>
          </div>

          <label className="space-y-2 text-sm font-medium">
            CPF
            <input
              className="h-11 w-full rounded-md border border-border bg-background px-3 outline-none focus:ring-2 focus:ring-primary/30"
              inputMode="numeric"
              maxLength={11}
              onChange={(event) => setCpf(event.target.value.replace(/\D/g, ""))}
              value={cpf}
            />
          </label>

          <label className="space-y-2 text-sm font-medium">
            Senha
            <input
              className="h-11 w-full rounded-md border border-border bg-background px-3 outline-none focus:ring-2 focus:ring-primary/30"
              onChange={(event) => setPassword(event.target.value)}
              type="password"
              value={password}
            />
          </label>

          {error ? <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p> : null}

          <Button type="submit" className="w-full" disabled={isLoading || cpf.length !== 11 || password.length === 0}>
            {isLoading ? "Entrando..." : "Entrar"}
            <ArrowRight className="h-4 w-4" />
          </Button>
        </form>
      </section>
    </main>
  );
}
