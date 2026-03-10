"""HuggingFace Spaces CLI commands for Terradev."""
import os, json, asyncio, click, sys


def _get_hf_token(TerradevAPI):
    token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
    if not token:
        token = TerradevAPI()._provider_creds('huggingface').get('api_key')
    if not token:
        print("\u274c HF_TOKEN not set. export HF_TOKEN=your_token")
        sys.exit(2)
    return token


def _hf_api(method, path, token, body=None):
    import urllib.request, urllib.error
    url = f"https://huggingface.co/api{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header('Authorization', f'Bearer {token}')
    if data:
        req.add_header('Content-Type', 'application/json')
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode()
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as e:
        raise Exception(f"HF API {e.code}: {e.read().decode()[:200]}")


def register_hf_spaces_commands(cli, TerradevAPI):
    @cli.group('hf-spaces')
    def hf_spaces():
        """HuggingFace Spaces — create, list, manage, delete."""
        pass

    @hf_spaces.command('create')
    @click.argument('space_name')
    @click.option('--model-id', required=True)
    @click.option('--hardware', default='cpu-basic',
                  type=click.Choice(['cpu-basic','cpu-upgrade','t4-medium','a10g-large','a100-large']))
    @click.option('--sdk', default='gradio', type=click.Choice(['gradio','streamlit','docker']))
    @click.option('--private', is_flag=True)
    @click.option('--template', type=click.Choice(['llm','embedding','image']), default=None)
    def create(space_name, model_id, hardware, sdk, private, template):
        """Create a new Space with auto-generated app."""
        from core.hf_spaces import HFSpacesDeployer, HFSpaceConfig, HFSpaceTemplates
        token = _get_hf_token(TerradevAPI)
        deployer = HFSpacesDeployer(token)
        if template:
            fn = {'llm': HFSpaceTemplates.get_llm_template, 'embedding': HFSpaceTemplates.get_embedding_template,
                  'image': HFSpaceTemplates.get_image_model_template}[template]
            config = fn(model_id, space_name)
            config.hardware = hardware; config.sdk = sdk; config.private = private
        else:
            config = HFSpaceConfig(name=space_name, model_id=model_id, hardware=hardware, sdk=sdk, private=private)
        print(f"Creating Space '{space_name}' on {hardware}...")
        r = asyncio.run(deployer.create_space(config))
        print(f"\u2705 {r['space_url']}" if r['status'] == 'created' else f"\u274c {r['error']}")

    @hf_spaces.command('list')
    @click.option('--author', default=None)
    @click.option('--limit', '-n', default=20, type=int)
    @click.option('--format', '-f', 'fmt', type=click.Choice(['json','text']), default='text')
    def list_spaces(author, limit, fmt):
        """List HuggingFace Spaces."""
        token = _get_hf_token(TerradevAPI)
        path = f'/spaces?limit={limit}' + (f'&author={author}' if author else '')
        spaces = _hf_api('GET', path, token)
        if fmt == 'json':
            print(json.dumps(spaces[:limit], indent=2, default=str)); return
        for s in (spaces or [])[:limit]:
            print(f"  {s.get('id','?'):<44} sdk={s.get('sdk','?')}")

    @hf_spaces.command('info')
    @click.argument('space_id')
    @click.option('--format', '-f', 'fmt', type=click.Choice(['json','text']), default='text')
    def info(space_id, fmt):
        """Get Space details."""
        token = _get_hf_token(TerradevAPI)
        d = _hf_api('GET', f'/spaces/{space_id}', token)
        if fmt == 'json':
            print(json.dumps(d, indent=2, default=str)); return
        rt = d.get('runtime') or {}
        hw = rt.get('hardware',{}).get('current','?') if isinstance(rt, dict) else '?'
        print(f"  {d.get('id','?')}  sdk={d.get('sdk','?')}  hw={hw}  private={d.get('private',False)}")

    @hf_spaces.command('delete')
    @click.argument('space_id')
    @click.option('--yes', '-y', is_flag=True)
    def delete(space_id, yes):
        """Delete a Space."""
        if not yes: click.confirm(f"Delete '{space_id}'?", abort=True)
        _hf_api('DELETE', f'/repos/delete', _get_hf_token(TerradevAPI), body={'type':'space','name':space_id})
        print(f"\u2705 Deleted {space_id}")

    @hf_spaces.command('restart')
    @click.argument('space_id')
    def restart(space_id):
        """Restart a Space (factory reboot)."""
        _hf_api('POST', f'/spaces/{space_id}/restart', _get_hf_token(TerradevAPI))
        print(f"\u2705 Restarting {space_id}")

    @hf_spaces.command('pause')
    @click.argument('space_id')
    def pause(space_id):
        """Pause a running Space (stops billing)."""
        _hf_api('POST', f'/spaces/{space_id}/pause', _get_hf_token(TerradevAPI))
        print(f"\u2705 Paused {space_id}")

    @hf_spaces.command('resume')
    @click.argument('space_id')
    def resume(space_id):
        """Resume a paused Space."""
        _hf_api('POST', f'/spaces/{space_id}/resume', _get_hf_token(TerradevAPI))
        print(f"\u2705 Resumed {space_id}")

    @hf_spaces.command('hardware')
    @click.argument('space_id')
    @click.option('--set', 'new_hw', default=None,
                  type=click.Choice(['cpu-basic','cpu-upgrade','t4-medium','a10g-large','a100-large']))
    def hardware(space_id, new_hw):
        """Show or change hardware tier."""
        token = _get_hf_token(TerradevAPI)
        if new_hw:
            _hf_api('POST', f'/spaces/{space_id}/hardware', token, body={'flavor': new_hw})
            print(f"\u2705 Hardware \u2192 {new_hw}")
        else:
            d = _hf_api('GET', f'/spaces/{space_id}', token)
            hw = (d.get('runtime') or {}).get('hardware', {})
            print(f"  Current: {hw.get('current','?')}  Requested: {hw.get('requested','?')}")

    @hf_spaces.command('logs')
    @click.argument('space_id')
    def logs(space_id):
        """Show Space build/run logs."""
        token = _get_hf_token(TerradevAPI)
        for kind in ['build', 'run']:
            try:
                d = _hf_api('GET', f'/spaces/{space_id}/logs/{kind}', token)
                lines = d if isinstance(d, list) else d.get('logs', d.get('lines', []))
                for line in (lines or [])[-30:]:
                    print(f"  [{kind}] {line}")
            except Exception:
                pass
