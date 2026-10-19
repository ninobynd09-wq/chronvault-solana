//! ChronVault on Solana: a one-way time lock for native SOL (added 2026-10).
//!
//! Same promise as the EVM TimeLock contracts: no owner, no admin, no pause, no early exit, and no
//! upgrade path (the program is deployed with `--final`, so nobody holds an upgrade key).
//!
//! Each lock is two addresses, both derived from the lock's terms, so the deposit address is known
//! before anything exists on chain:
//!   vault = PDA("vault", beneficiary, duration, salt)  plain system account that holds the SOL.
//!                                                     Anyone can send to it, before or after Init.
//!   state = PDA("state", vault)                        this program's record: beneficiary, unlock time.
//!
//! Init (open to anyone, like the EVM factory's deploy): writes the state with
//! unlock_time = now + duration. The clock starts here, when the deposit arrived. The terms are
//! fixed by the vault address itself, so nobody can start a funded vault with other terms.
//!
//! Withdraw (open to anyone): once the cluster clock reaches unlock_time, moves the vault's entire
//! balance to the beneficiary written in the state. Nothing else can move it, ever.
//!
//! Written to stay small (deployment rent is paid per byte): no_std, no allocator,
//! a lazy entrypoint that only reads the four accounts used, no panicking code paths, no floating
//! point, no log messages, and the three system-program calls built by hand. Behaviour, seeds, state
//! layout and error codes are identical to the first version.
#![no_std]

use core::mem::MaybeUninit;
use pinocchio::{
    account_info::AccountInfo,
    cpi::invoke_signed_unchecked,
    entrypoint::{InstructionContext, MaybeAccount},
    instruction::{Account, AccountMeta, Instruction, Seed, Signer},
    no_allocator,
    pubkey::Pubkey,
    sysvars::{clock::Clock, rent::Rent},
};

no_allocator!();

/// Errors are the runtime's raw u64 codes, exactly the values the first version returned: a custom
/// code n is n, a builtin error k is k << 32 (pinocchio's constants). Returning them directly avoids
/// the big enum-to-number table the generic entrypoint carries.
type ProgramResult = Result<(), u64>;
const MISSING_REQUIRED_SIGNATURES: u64 = 8 << 32;
const NOT_ENOUGH_ACCOUNT_KEYS: u64 = 11 << 32;
const INVALID_INSTRUCTION_DATA: u64 = 3 << 32;

/// Program entrypoint (what `lazy_program_entrypoint!` expands to, minus the error-enum conversion).
#[no_mangle]
pub unsafe extern "C" fn entrypoint(input: *mut u8) -> u64 {
    match process(InstructionContext::new_unchecked(input)) {
        Ok(()) => 0,
        Err(code) => code,
    }
}

/// Nothing in this program can panic (every check returns an error instead); if something ever did,
/// abort without pulling in any message formatting.
#[cfg(target_os = "solana")]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    unsafe { pinocchio::syscalls::abort() }
}

pub const MAGIC: &[u8; 8] = b"CVLOCK01";
/// magic 8 | beneficiary 32 | unlock_time i64 8 | duration u64 8 | salt 32 | vault_bump 1 | state_bump 1
pub const STATE_LEN: usize = 90;

pub const ERR_LOCKED: u32 = 1; // too early
pub const ERR_EMPTY: u32 = 2; // nothing in the vault
pub const ERR_BAD_ACCOUNT: u32 = 3; // an account isn't the one these terms derive
pub const ERR_ALREADY_INIT: u32 = 4; // this lock's clock was already started
pub const ERR_BAD_DATA: u32 = 5; // malformed instruction or state
pub const ERR_OVERFLOW: u32 = 6;

const SYSTEM: Pubkey = [0u8; 32];

#[inline(always)]
fn fail(code: u32) -> u64 {
    code as u64
}


/// The next account, refusing duplicates (none of this program's accounts may repeat; a repeat
/// could never pass the key checks below anyway).
#[inline(always)]
fn next(ctx: &mut InstructionContext) -> Result<AccountInfo, u64> {
    if ctx.remaining() == 0 {
        return Err(NOT_ENOUGH_ACCOUNT_KEYS);
    }
    match ctx.next_account().map_err(|_| NOT_ENOUGH_ACCOUNT_KEYS)? {
        MaybeAccount::Account(a) => Ok(a),
        MaybeAccount::Duplicated(_) => Err(fail(ERR_BAD_ACCOUNT)),
    }
}

/// Fixed-size copy out of a slice whose length was already checked.
#[inline(always)]
fn arr<const N: usize>(s: &[u8], at: usize) -> [u8; N] {
    let mut o = [0u8; N];
    o.copy_from_slice(&s[at..at + N]);
    o
}

fn process(mut ctx: InstructionContext) -> ProgramResult {
    let a = next(&mut ctx)?;
    let b = next(&mut ctx)?;
    let c = next(&mut ctx)?;
    let sys = next(&mut ctx)?;
    // Extra accounts after the four are ignored, as before.
    while ctx.remaining() > 0 {
        let _ = ctx.next_account(); // counts down `remaining`
    }
    // every account has been read, so these can't fail
    let data = ctx.instruction_data().map_err(|_| INVALID_INSTRUCTION_DATA)?;
    let program_id = ctx.program_id().map_err(|_| INVALID_INSTRUCTION_DATA)?;
    if sys.key() != &SYSTEM {
        return Err(fail(ERR_BAD_ACCOUNT));
    }
    match data.first() {
        Some(0) if data.len() == 73 => init(program_id, &a, &b, &c, &sys, data),
        Some(1) if data.len() == 1 => withdraw(program_id, &a, &b, &c, &sys),
        _ => Err(fail(ERR_BAD_DATA)),
    }
}

/// One system-program call; `data` is the complete instruction (u32 kind + arguments). `first` always
/// signs (as a PDA when `seeds` is given); `second`, if any, is writable and signs only for
/// CreateAccount (`second_signs`).
#[inline(never)]
fn system_call(data: &[u8], first: &AccountInfo, second: Option<&AccountInfo>, second_signs: bool, seeds: &[Seed]) -> ProgramResult {
    let other = match second { Some(a) => a, None => first };
    let metas = [
        AccountMeta { pubkey: first.key(), is_writable: true, is_signer: true },
        AccountMeta { pubkey: other.key(), is_writable: true, is_signer: second_signs },
    ];
    let infos = [Account::from(first), Account::from(other)];
    let n = if second.is_some() { 2 } else { 1 };
    let ix = Instruction { program_id: &SYSTEM, accounts: unsafe { metas.get_unchecked(..n) }, data };
    let signer = [Signer::from(seeds)];
    // No account data or lamports are borrowed by this program while the call runs.
    unsafe { invoke_signed_unchecked(&ix, infos.get_unchecked(..n), if seeds.is_empty() { &[] } else { &signer }) };
    Ok(())
}

/// [kind u32 LE][value u64 LE]: Transfer (2) and Allocate (8).
#[inline(always)]
fn ix_u64(kind: u8, v: u64) -> [u8; 12] {
    let mut b = [0u8; 12];
    b[0] = kind;
    b[4..12].copy_from_slice(&v.to_le_bytes());
    b
}

/// The PDA and bump for `seeds` (same syscall find_program_address uses; like it, a failure to find a
/// bump, which is statistically impossible, aborts the program).
#[inline(never)]
fn find(seeds: &[&[u8]], program_id: &Pubkey) -> (Pubkey, u8) {
    let mut out = MaybeUninit::<Pubkey>::uninit();
    let mut bump = u8::MAX;
    let r = unsafe {
        pinocchio::syscalls::sol_try_find_program_address(
            seeds as *const _ as *const u8, seeds.len() as u64, program_id as *const _ as *const u8,
            out.as_mut_ptr() as *mut u8, &mut bump as *mut u8)
    };
    if r != 0 {
        unsafe { pinocchio::syscalls::abort() }
    }
    (unsafe { out.assume_init() }, bump)
}

/// The address the runtime derives from `seeds` + `program_id` (same syscall as
/// create_program_address); any failure is BadAccount, as before.
#[inline(never)]
fn pda(seeds: &[&[u8]], program_id: &Pubkey) -> Result<Pubkey, u64> {
    let mut out = MaybeUninit::<Pubkey>::uninit();
    let r = unsafe {
        pinocchio::syscalls::sol_create_program_address(
            seeds as *const _ as *const u8, seeds.len() as u64, program_id as *const _ as *const u8, out.as_mut_ptr() as *mut u8)
    };
    if r != 0 {
        return Err(fail(ERR_BAD_ACCOUNT));
    }
    Ok(unsafe { out.assume_init() })
}

/// A sysvar read straight from the runtime (same syscall Sysvar::get uses); errors keep the runtime's code.
#[inline(always)]
fn sysvar<T>(call: unsafe extern "C" fn(*mut u8) -> u64) -> Result<T, u64> {
    let mut v = MaybeUninit::<T>::uninit();
    let r = unsafe { call(v.as_mut_ptr() as *mut u8) };
    if r != 0 {
        return Err(r);
    }
    Ok(unsafe { v.assume_init() })
}

#[inline(always)]
fn clock_now() -> Result<i64, u64> {
    Ok(sysvar::<Clock>(pinocchio::syscalls::sol_get_clock_sysvar)?.unix_timestamp)
}

fn state_rent() -> Result<u64, u64> {
    let rent: Rent = sysvar(pinocchio::syscalls::sol_get_rent_sysvar)?;
    // Range limits far above any real value (today 3,480 lamports per byte-year, threshold 2.0) under
    // which these products cannot overflow: (90+128) * 2^32 * 2^20 < 2^60.
    if rent.lamports_per_byte_year >> 32 != 0 {
        return Err(fail(ERR_OVERFLOW));
    }
    let base = (STATE_LEN as u64 + 128) * rent.lamports_per_byte_year;
    #[allow(deprecated)]
    let bits = u64::from_le_bytes(rent.exemption_threshold.to_le_bytes());
    // Multiplier = the threshold rounded UP to a power of two: exactly 1 or 2 for the values Solana uses
    // (1.0, 2.0), and never below the real threshold otherwise.
    let exp = ((bits >> 52) & 0x7ff) as i64 - 1023;
    let frac = bits & ((1u64 << 52) - 1);
    if bits >> 63 != 0 || exp > 19 {
        return Err(fail(ERR_OVERFLOW));
    }
    let mult = if exp < 0 { 1 } else if frac == 0 { 1u64 << exp } else { 1u64 << (exp + 1) };
    Ok(base * mult)
}

/// Accounts: 0 payer (signer, writable), 1 vault, 2 state (writable), 3 system program.
/// Data: 0 | beneficiary 32 | duration u64 LE | salt 32  (length checked by the caller: 73 bytes).
fn init(
    program_id: &Pubkey,
    payer: &AccountInfo,
    vault: &AccountInfo,
    state: &AccountInfo,
    _sys: &AccountInfo,
    data: &[u8],
) -> ProgramResult {
    let terms = &data[1..73]; // beneficiary 32 | duration 8 | salt 32, exactly as stored in the state
    let beneficiary = &terms[0..32];
    let dur_bytes = &terms[32..40];
    let salt = &terms[40..72];
    let duration = u64::from_le_bytes(arr(dur_bytes, 0));
    if duration > i64::MAX as u64 {
        return Err(fail(ERR_OVERFLOW));
    }
    if !payer.is_signer() {
        return Err(MISSING_REQUIRED_SIGNATURES);
    }
    let (vault_key, vault_bump) = find(&[b"vault", beneficiary, dur_bytes, salt], program_id);
    if vault.key() != &vault_key {
        return Err(fail(ERR_BAD_ACCOUNT));
    }
    // The vault is and stays a plain system account (only this program could ever sign for it).
    if !vault.is_owned_by(&SYSTEM) || !vault.data_is_empty() {
        return Err(fail(ERR_BAD_ACCOUNT));
    }
    let (state_key, state_bump) = find(&[b"state", &vault_key], program_id);
    if state.key() != &state_key {
        return Err(fail(ERR_BAD_ACCOUNT));
    }
    if state.is_owned_by(program_id) {
        return Err(fail(ERR_ALREADY_INIT));
    }
    if !state.is_owned_by(&SYSTEM) || !state.data_is_empty() {
        return Err(fail(ERR_BAD_ACCOUNT));
    }

    let need = state_rent()?;
    let bump = [state_bump];
    let seeds = [Seed::from(b"state"), Seed::from(&vault_key), Seed::from(&bump)];
    let len = (STATE_LEN as u64).to_le_bytes();
    if state.lamports() == 0 {
        // CreateAccount: kind 0 | lamports | space | owner
        let mut ix = [0u8; 52];
        ix[4..12].copy_from_slice(&need.to_le_bytes());
        ix[12..20].copy_from_slice(&len);
        ix[20..52].copy_from_slice(program_id);
        system_call(&ix, payer, Some(state), true, &seeds)?;
    } else {
        // Somebody sent SOL to the state address first, which would make CreateAccount fail and
        // block this lock forever. Top it up to rent-exempt, then allocate and assign it instead.
        let short = need.saturating_sub(state.lamports());
        if short > 0 {
            system_call(&ix_u64(2, short), payer, Some(state), false, &[])?;
        }
        system_call(&ix_u64(8, STATE_LEN as u64), state, None, false, &seeds)?;
        let mut assign = [0u8; 36];
        assign[0] = 1;
        assign[4..36].copy_from_slice(program_id);
        system_call(&assign, state, None, false, &seeds)?;
    }

    let now = clock_now()?;
    let unlock = now.checked_add(duration as i64).ok_or(fail(ERR_OVERFLOW))?;
    // The account was just created/allocated by this instruction and nothing else holds its data.
    let d = unsafe { state.borrow_mut_data_unchecked() };
    if d.len() != STATE_LEN {
        return Err(fail(ERR_BAD_DATA));
    }
    d[0..8].copy_from_slice(MAGIC);
    d[8..40].copy_from_slice(beneficiary);
    d[40..48].copy_from_slice(&unlock.to_le_bytes());
    d[48..88].copy_from_slice(&terms[32..72]); // duration | salt
    d[88] = vault_bump;
    d[89] = state_bump;
    Ok(())
}

/// Accounts: 0 vault (writable), 1 state, 2 beneficiary (writable), 3 system program. Anyone may send it.
fn withdraw(
    program_id: &Pubkey,
    vault: &AccountInfo,
    state: &AccountInfo,
    beneficiary: &AccountInfo,
    _sys: &AccountInfo,
) -> ProgramResult {
    if !state.is_owned_by(program_id) {
        return Err(fail(ERR_BAD_ACCOUNT));
    }
    // Read-only view of the state; this instruction never writes it.
    let d = unsafe { state.borrow_data_unchecked() };
    if d.len() != STATE_LEN || &d[0..8] != MAGIC {
        return Err(fail(ERR_BAD_DATA));
    }
    let ben_key = &d[8..40];
    let unlock = i64::from_le_bytes(arr(d, 40));
    let dur_bytes = &d[48..56];
    let salt = &d[56..88];
    let vbump = &d[88..89];
    let sbump = &d[89..90];
    // The state must belong to this vault, and the vault must be the one these terms derive.
    let expected_vault = pda(&[b"vault", ben_key, dur_bytes, salt, vbump], program_id)?;
    if vault.key() != &expected_vault {
        return Err(fail(ERR_BAD_ACCOUNT));
    }
    let expected_state = pda(&[b"state", vault.key(), sbump], program_id)?;
    if state.key() != &expected_state {
        return Err(fail(ERR_BAD_ACCOUNT));
    }
    // Money only ever goes to the beneficiary written at Init.
    if &beneficiary.key()[..] != ben_key || !vault.is_writable() || !beneficiary.is_writable() {
        return Err(fail(ERR_BAD_ACCOUNT));
    }
    if clock_now()? < unlock {
        return Err(fail(ERR_LOCKED));
    }
    let amount = vault.lamports();
    if amount == 0 {
        return Err(fail(ERR_EMPTY));
    }
    let seeds = [Seed::from(b"vault"), Seed::from(ben_key), Seed::from(dur_bytes), Seed::from(salt), Seed::from(vbump)];
    system_call(&ix_u64(2, amount), vault, Some(beneficiary), false, &seeds)
}
