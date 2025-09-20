---
title: Local Server (Server Mode)
parent: Instructions
nav_order: 2
has_children: true
---

# Local Server (Server Mode)

## Activate "Server Mode" 

simply open PowerShell and enter:

```bash
flm serve llama3.2:1b
```

You can choose to change the server port (default is 11434) by going to **System Properties** → **Environment Variables**, then modifying the value of `FLM_SERVE_PORT`.

> ⚠️ **Be cautious**: If you update this value, be sure to change any higher-level port settings in your application as well to ensure everything works correctly.

## Set Context Length at Launch

The default context length for each model can be found [here](https://docs.fastflowlm.com/models/). 

To change it at launch, in PowerShell, run:

```bash
flm serve llama3.2:1b --ctx-len 8192
```

> Internally, FLM enforces a minimum context length of 512. If you specify a smaller value, it will automatically be adjusted up to 512.  

## Set Request Queue in Server Mode

Since v0.9.10, FLM adds a request queue in server mode to prevent overload under high traffic.  
This keeps processing stable and orderly when multiple requests arrive.

- **Default:** 10  
- **Change with:** `--q-len` (or `-q`)  

To change it at launch, in PowerShell, run:
  
```powershell
flm serve llama3.2:1b --q-len 20
```

## Customizable Socket Connections in Server Mode

Set the maximum number of concurrent socket connections to control network resource usage.
👉 Recommended: set sockets **equal to or greater than** the queue length.

- **Default:** 10  
- **Change with:** `--socket` (or `-s`)  

To change it at launch, in PowerShell, run:

```powershell
flm serve llama3.2:1b --socket 20
```