import getpass

key = getpass.getpass("Paste your Polymarket private key and press Enter: ")
with open(".env", "w") as f:
    f.write(f"POLYMARKET_PRIVATE_KEY={key}\n")
    f.write("POLYMARKET_WALLET=0xc4Cd289962f07720Ca3A0FB384917Db2530cf58D\n")
print("Saved to .env successfully!")
