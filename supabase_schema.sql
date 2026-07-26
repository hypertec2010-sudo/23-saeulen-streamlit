-- v28.0 migration bridge: one JSON document per user and namespace.
-- Run this once in the Supabase SQL Editor.

create table if not exists public.app_state (
    user_id text not null,
    namespace text not null,
    payload jsonb not null default '{}'::jsonb,
    updated_at timestamptz not null default now(),
    primary key (user_id, namespace)
);

create index if not exists app_state_updated_at_idx
    on public.app_state (updated_at desc);

create or replace function public.touch_app_state_updated_at()
returns trigger
language plpgsql
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

drop trigger if exists app_state_touch_updated_at on public.app_state;
create trigger app_state_touch_updated_at
before update on public.app_state
for each row execute function public.touch_app_state_updated_at();

alter table public.app_state enable row level security;

-- The Streamlit server uses the service-role key from Secrets.
-- No public anon policy is created intentionally.
