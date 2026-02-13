"""
Polymarket Bot Setup Wizard.

Walks through:
  1. Install dependencies (py-clob-client, web3)
  2. Create or import a Polygon wallet
  3. Derive Polymarket API credentials
  4. Fund the wallet with USDC on Polygon
  5. Set USDC allowance for Polymarket's CTF Exchange
  6. Verify connectivity

Usage:
    python -m polymarket --setup
"""

import getpass
import json
import os
import sys
from pathlib import Path

from .config import POLYMARKET


def _check_dependencies() -> bool:
    """Check that required packages are installed."""
    missing = []

    try:
        import py_clob_client  # noqa: F401
    except ImportError:
        missing.append("py-clob-client")

    try:
        import web3  # noqa: F401
    except ImportError:
        missing.append("web3")

    if missing:
        print()
        print("  Missing dependencies:")
        for pkg in missing:
            print(f"    - {pkg}")
        print()
        print("  Install them with:")
        print(f"    pip install {' '.join(missing)}")
        print()
        return False

    return True


def _create_or_import_wallet() -> tuple[str, str]:
    """
    Create a new Polygon wallet or import an existing private key.

    Returns (private_key_hex, wallet_address).
    """
    print()
    print("  STEP 1: Wallet Setup")
    print("  " + "-" * 40)
    print()
    print("  Polymarket runs on Polygon. You need a wallet with:")
    print("    - A private key (for signing orders)")
    print("    - USDC on Polygon (for placing trades)")
    print()
    print("  Options:")
    print("    1) Import existing private key")
    print("    2) Generate a new wallet")
    print()

    choice = input("  Enter 1 or 2: ").strip()

    if choice == "2":
        from eth_account import Account
        acct = Account.create()
        private_key = acct.key.hex()
        if private_key.startswith("0x"):
            private_key = private_key[2:]
        address = acct.address
        print()
        print(f"  New wallet created!")
        print(f"  Address:     {address}")
        print(f"  Private key: {private_key[:8]}...{private_key[-8:]}")
        print()
        print("  IMPORTANT: Save this private key somewhere safe!")
        print("  If you lose it, you lose access to any funds in this wallet.")
        return private_key, address

    else:
        print()
        private_key = getpass.getpass("  Enter your private key (hex, no 0x prefix): ").strip()
        if private_key.startswith("0x"):
            private_key = private_key[2:]

        # Validate and derive address
        try:
            from eth_account import Account
            acct = Account.from_key(bytes.fromhex(private_key))
            address = acct.address
            print(f"  Wallet address: {address}")
            return private_key, address
        except Exception as e:
            print(f"  Invalid private key: {e}")
            sys.exit(1)


def _derive_api_creds(private_key: str, chain_id: int = 137) -> dict:
    """Derive Polymarket CLOB API credentials from wallet."""
    print()
    print("  STEP 2: Derive API Credentials")
    print("  " + "-" * 40)
    print()
    print("  Polymarket uses API keys derived from your wallet signature.")
    print("  This signs a message with your private key to generate creds.")
    print()

    from py_clob_client.client import ClobClient

    clob_url = POLYMARKET.get("clob_url", "https://clob.polymarket.com")

    client = ClobClient(clob_url, key=private_key, chain_id=chain_id)
    creds = client.create_or_derive_api_creds()

    if hasattr(creds, "api_key"):
        result = {
            "api_key": creds.api_key,
            "api_secret": creds.api_secret,
            "api_passphrase": creds.api_passphrase,
        }
    elif isinstance(creds, dict):
        result = creds
    else:
        print(f"  Unexpected credential format: {type(creds)}")
        sys.exit(1)

    print(f"  API key:        {result['api_key'][:12]}...")
    print(f"  API secret:     {result['api_secret'][:12]}...")
    print(f"  API passphrase: {result['api_passphrase'][:12]}...")
    print()
    print("  Credentials derived successfully.")

    return result


def _save_env(private_key: str, address: str, creds: dict, funder: str = ""):
    """Save credentials to .env file."""
    print()
    print("  STEP 3: Save Credentials")
    print("  " + "-" * 40)

    env_path = Path(".env")

    lines = [
        f"POLYMARKET_PRIVATE_KEY={private_key}",
        f"POLYMARKET_WALLET_ADDRESS={address}",
        f"POLYMARKET_API_KEY={creds['api_key']}",
        f"POLYMARKET_API_SECRET={creds['api_secret']}",
        f"POLYMARKET_API_PASSPHRASE={creds['api_passphrase']}",
        f"POLYMARKET_CHAIN_ID=137",
    ]

    if funder:
        lines.append(f"POLYMARKET_FUNDER={funder}")

    # Append to existing .env or create new
    with open(env_path, "a") as f:
        f.write("\n# Polymarket Bot Credentials\n")
        for line in lines:
            f.write(line + "\n")

    print(f"  Credentials saved to {env_path.resolve()}")
    print(f"  This file is in .gitignore — never commit it.")


def _print_funding_instructions(address: str):
    """Print instructions for funding the wallet."""
    print()
    print("  STEP 4: Fund Your Wallet")
    print("  " + "-" * 40)
    print()
    print("  Your wallet needs USDC on the Polygon network to trade.")
    print()
    print(f"  Your wallet address: {address}")
    print()
    print("  Option A — Bridge from Ethereum:")
    print("    1. Go to https://wallet.polygon.technology/polygon/bridge")
    print("    2. Bridge USDC from Ethereum to Polygon")
    print("    3. You'll also need a small amount of MATIC for gas")
    print()
    print("  Option B — Buy directly on Polygon:")
    print("    1. Use an exchange that supports Polygon withdrawals")
    print("       (Binance, Coinbase, Kraken all support Polygon USDC)")
    print("    2. Withdraw USDC to your wallet address on Polygon network")
    print()
    print("  Option C — Fund via Polymarket directly:")
    print("    1. Go to https://polymarket.com")
    print("    2. Connect your wallet or import your private key")
    print("    3. Use their deposit flow (supports credit card, crypto)")
    print()
    print("  Recommended starting amount: $50-100 for paper testing,")
    print("  $500+ for live trading with the default risk limits.")
    print()
    print("  You also need ~0.1 MATIC for gas fees on Polygon.")
    print("  Most exchanges let you withdraw MATIC to Polygon directly.")
    print()


def _print_allowance_instructions():
    """Print USDC allowance instructions."""
    print()
    print("  STEP 5: USDC Allowance")
    print("  " + "-" * 40)
    print()
    print("  Before the bot can trade, your wallet must approve Polymarket's")
    print("  CTF Exchange contract to spend your USDC.")
    print()
    print("  This happens automatically on the first trade if you're using")
    print("  the py-clob-client SDK. The SDK will prompt for approval.")
    print()
    print("  Alternatively, approve manually at:")
    print("    https://polygonscan.com/token/0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174#writeContract")
    print("    Function: approve(spender, amount)")
    print("    Spender:  0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E  (CTF Exchange)")
    print("    Amount:   Max uint256 for unlimited approval")
    print()


def _verify_connection(private_key: str, creds: dict):
    """Verify that we can connect to Polymarket."""
    print()
    print("  STEP 6: Verify Connection")
    print("  " + "-" * 40)
    print()

    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import ApiCreds

    clob_url = POLYMARKET.get("clob_url", "https://clob.polymarket.com")
    chain_id = POLYMARKET.get("chain_id", 137)

    api_creds = ApiCreds(
        api_key=creds["api_key"],
        api_secret=creds["api_secret"],
        api_passphrase=creds["api_passphrase"],
    )

    client = ClobClient(
        clob_url,
        key=private_key,
        chain_id=chain_id,
        creds=api_creds,
    )

    # Test public endpoint
    print("  Testing public API... ", end="", flush=True)
    try:
        import urllib.request
        import urllib.error
        req = urllib.request.Request(
            f"{clob_url}/midpoint?token_id=0x0000000000000000000000000000000000000000000000000000000000000001",
            headers={"Accept": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10)
        print("OK")
    except urllib.error.HTTPError:
        # 404 is fine — means the API is reachable
        print("OK (API reachable)")
    except Exception as e:
        print(f"FAILED: {e}")

    # Test authenticated endpoint
    print("  Testing authenticated API... ", end="", flush=True)
    try:
        result = client.get_orders()
        print("OK")
    except Exception as e:
        err_str = str(e)
        if "no orders" in err_str.lower() or "[]" in err_str:
            print("OK (no open orders)")
        else:
            print(f"WARNING: {e}")
            print("  (This may be normal if the wallet has no trading history)")

    print()


def run_setup():
    """Run the full setup wizard."""
    print()
    print("=" * 60)
    print("  POLYMARKET BOT — SETUP WIZARD")
    print("=" * 60)
    print()
    print("  This wizard will help you connect the bot to Polymarket.")
    print("  You'll need:")
    print("    - A Polygon wallet (or we'll create one)")
    print("    - USDC on Polygon for trading")
    print()

    # Step 0: Check dependencies
    if not _check_dependencies():
        print("  Install the missing packages and run --setup again.")
        return

    # Step 1: Wallet
    private_key, address = _create_or_import_wallet()

    # Step 2: API creds
    chain_id = POLYMARKET.get("chain_id", 137)
    try:
        creds = _derive_api_creds(private_key, chain_id)
    except Exception as e:
        print(f"  Failed to derive API credentials: {e}")
        print("  Make sure py-clob-client is installed: pip install py-clob-client")
        return

    # Step 3: Save to .env
    _save_env(private_key, address, creds)

    # Step 4: Funding instructions
    _print_funding_instructions(address)

    # Step 5: Allowance
    _print_allowance_instructions()

    # Step 6: Verify
    try:
        _verify_connection(private_key, creds)
    except Exception as e:
        print(f"  Verification failed: {e}")
        print("  You can still proceed — verification is optional.")

    # Done
    print("=" * 60)
    print("  SETUP COMPLETE")
    print("=" * 60)
    print()
    print("  Next steps:")
    print("    1. Fund your wallet with USDC on Polygon (see Step 4 above)")
    print("    2. Start with paper trading to test:")
    print("       python -m polymarket --paper --loop")
    print("    3. View available markets:")
    print("       python -m polymarket --markets")
    print("    4. Run a single strategy:")
    print("       python -m polymarket --strategy high_prob_grinder --paper")
    print("    5. When ready for live trading:")
    print("       python -m polymarket --live --loop")
    print()
    print("  Configuration: polymarket/config.py")
    print("  Credentials:   .env (never commit this)")
    print("  Trade log:     polymarket/trades.jsonl")
    print()
