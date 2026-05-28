import {
  ArrowRight,
  BarChart3,
  Camera,
  Eye,
  EyeOff,
  Lock,
  LockKeyhole,
  ScanFace,
  Sparkles,
  UserRound,
} from "lucide-react";
import { useState } from "react";

import classAiClassroomHero from "@/assets/classai-classroom-hero.png";
import { Button } from "@/components/ui/Button";
import { useAuthStore } from "@/stores/authStore";

const featureCards = [
  {
    icon: Camera,
    title: "Monitoramento comportamental",
    subtitle: "em tempo real",
  },
  {
    icon: BarChart3,
    title: "Learning Analytics",
    subtitle: "para decisões pedagógicas",
  },
  {
    icon: ScanFace,
    title: "Reconhecimento inteligente",
    subtitle: "com visão computacional",
  },
];

function ClassAiMark({ compact = false }: { compact?: boolean }) {
  return (
    <div className={compact ? "flex items-center gap-4" : "flex items-center gap-5"}>
      <div
        className={
          compact
            ? "flex h-14 w-14 items-center justify-center rounded-lg bg-[#06172f] text-sky-300 shadow-[0_18px_45px_rgba(14,116,202,0.28)] ring-1 ring-sky-300/15"
            : "flex h-12 w-12 items-center justify-center text-sky-300"
        }
      >
        <Sparkles className={compact ? "h-8 w-8" : "h-11 w-11"} strokeWidth={1.8} />
      </div>
      <div>
        <p className={compact ? "text-xl font-bold tracking-normal text-slate-950" : "text-4xl font-bold tracking-[0.12em] text-white"}>
          CLASS<span className="text-sky-400">AI</span>
        </p>
        <p
          className={
            compact
              ? "mt-1 text-sm font-medium text-slate-400"
              : "mt-2 text-xs font-semibold uppercase tracking-[0.34em] text-slate-300"
          }
        >
          {compact ? "Acesso seguro à plataforma" : "Inteligência Educacional"}
        </p>
      </div>
    </div>
  );
}

export function Login() {
  const login = useAuthStore((state) => state.login);
  const isLoading = useAuthStore((state) => state.isLoading);
  const error = useAuthStore((state) => state.error);
  const [cpf, setCpf] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [remember, setRemember] = useState(false);

  return (
    <main className="min-h-screen bg-white text-slate-950">
      <section className="grid min-h-screen lg:grid-cols-[58.5%_41.5%]">
        <aside className="relative hidden min-h-screen overflow-hidden bg-[#020817] text-white lg:block">
          <img
            alt="Sala de aula moderna monitorada por visão computacional"
            className="absolute inset-0 h-full w-full object-cover"
            src={classAiClassroomHero}
          />
          <div className="absolute inset-0 bg-gradient-to-r from-[#020817]/96 via-[#06224a]/70 to-[#020817]/30" />
          <div className="absolute inset-0 bg-gradient-to-t from-[#020817]/82 via-transparent to-[#020817]/40" />

          <div className="relative z-10 flex h-full min-h-screen flex-col px-14 py-16 xl:px-20">
            <ClassAiMark />

            <div className="mt-20 max-w-[620px]">
              <h1 className="text-[2.65rem] font-bold leading-[1.25] tracking-normal text-white xl:text-[3.45rem]">
                Inteligência educacional em tempo real para
                <span className="block text-sky-400">salas de aula modernas.</span>
              </h1>
              <p className="mt-8 max-w-[520px] text-lg leading-8 text-slate-300">
                Acompanhe o comportamento dos alunos, gere insights pedagógicos e transforme dados em melhores decisões.
              </p>
            </div>

            <div className="pointer-events-none absolute left-[58%] top-[45%] hidden -translate-x-1/2 xl:block">
              <div className="max-w-xs rounded-lg border border-white/12 bg-[#06172f]/36 p-4 text-sm leading-6 text-slate-200 shadow-2xl shadow-slate-950/20 backdrop-blur-md">
                <p className="text-xs font-semibold uppercase tracking-[0.18em] text-sky-200/80">Análise pedagógica</p>
                <p className="mt-2">Sinais de atenção e participação são avaliados de forma contínua e contextual.</p>
              </div>
            </div>

            <div className="mt-auto grid max-w-3xl grid-cols-3 gap-8">
              {featureCards.map((item, index) => {
                const Icon = item.icon;
                return (
                  <div className="relative min-h-[150px] pt-2" key={item.title}>
                    {index > 0 ? <div className="absolute -left-4 top-8 h-28 w-px bg-sky-300/20" /> : null}
                    <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-full border border-sky-300/30 bg-[#08284c]/70 text-sky-300 shadow-[0_14px_34px_rgba(2,8,23,0.28)] backdrop-blur-sm">
                      <Icon className="h-8 w-8" strokeWidth={1.8} />
                    </div>
                    <p className="max-w-[165px] text-base font-bold leading-6 text-white">{item.title}</p>
                    <p className="mt-1 max-w-[160px] text-base leading-6 text-slate-300">{item.subtitle}</p>
                  </div>
                );
              })}
            </div>
          </div>
        </aside>

        <div className="flex min-h-screen items-center justify-center bg-[#fbfcff] px-6 py-10 sm:px-10">
          <form
            className="w-full max-w-[430px]"
            onSubmit={(event) => {
              event.preventDefault();
              void login({ cpf, password });
            }}
          >
            <ClassAiMark compact />

            <div className="mt-14">
              <h2 className="text-[2rem] font-bold leading-tight tracking-normal text-slate-950">Acesse o painel</h2>
              <p className="mt-3 text-base text-slate-500">Entre com suas credenciais para continuar.</p>
            </div>

            <div className="mt-9 space-y-6">
              <label className="block space-y-2 text-sm font-bold text-slate-800">
                CPF
                <div className="flex h-14 items-center gap-4 rounded-lg border border-slate-200 bg-white px-4 shadow-sm transition focus-within:border-sky-400 focus-within:ring-4 focus-within:ring-sky-100">
                  <UserRound className="h-5 w-5 text-slate-400" strokeWidth={1.8} />
                  <input
                    className="h-full w-full bg-transparent text-base font-medium text-slate-900 outline-none placeholder:text-slate-400"
                    inputMode="numeric"
                    maxLength={11}
                    onChange={(event) => setCpf(event.target.value.replace(/\D/g, ""))}
                    placeholder="Digite seu CPF"
                    value={cpf}
                  />
                </div>
              </label>

              <label className="block space-y-2 text-sm font-bold text-slate-800">
                Senha
                <div className="flex h-14 items-center gap-4 rounded-lg border border-slate-200 bg-white px-4 shadow-sm transition focus-within:border-sky-400 focus-within:ring-4 focus-within:ring-sky-100">
                  <LockKeyhole className="h-5 w-5 text-slate-400" strokeWidth={1.8} />
                  <input
                    className="h-full w-full bg-transparent text-base font-medium text-slate-900 outline-none placeholder:text-slate-400"
                    onChange={(event) => setPassword(event.target.value)}
                    placeholder="Digite sua senha"
                    type={showPassword ? "text" : "password"}
                    value={password}
                  />
                  <button
                    aria-label={showPassword ? "Ocultar senha" : "Mostrar senha"}
                    className="text-slate-400 transition hover:text-slate-700"
                    onClick={() => setShowPassword((value) => !value)}
                    type="button"
                  >
                    {showPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                  </button>
                </div>
              </label>
            </div>

            <label className="mt-6 flex items-center gap-3 text-sm font-medium text-slate-500">
              <input
                checked={remember}
                className="h-4 w-4 rounded border-slate-300 text-sky-500 focus:ring-sky-300"
                onChange={(event) => setRemember(event.target.checked)}
                type="checkbox"
              />
              Lembrar-me
            </label>

            {error ? (
              <p className="mt-5 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm font-medium text-red-700">
                {error}
              </p>
            ) : null}

            <Button
              className="mt-7 h-14 w-full rounded-lg bg-gradient-to-r from-blue-600 to-sky-400 text-base font-bold text-white shadow-[0_18px_42px_rgba(14,116,202,0.24)] transition hover:-translate-y-0.5 hover:shadow-[0_22px_48px_rgba(14,116,202,0.32)]"
              disabled={isLoading || cpf.length !== 11 || password.length === 0}
              type="submit"
            >
              {isLoading ? (
                <>
                  <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/35 border-t-white" />
                  Entrando
                </>
              ) : (
                <>
                  Entrar
                  <ArrowRight className="h-5 w-5" />
                </>
              )}
            </Button>

            <div className="mt-9 border-t border-slate-200 pt-8 text-center">
              <button className="inline-flex items-center gap-2 text-sm font-bold text-blue-600 transition hover:text-sky-500" type="button">
                <Lock className="h-4 w-4 text-slate-400" />
                Esqueceu sua senha?
              </button>
            </div>

            <p className="mt-24 text-center text-sm text-slate-400">© 2025 CLASSAI. Todos os direitos reservados.</p>
          </form>
        </div>
      </section>
    </main>
  );
}
