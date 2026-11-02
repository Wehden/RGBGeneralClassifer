/* Shared volume wording for live, archived and comparison readings. */
(function(root){
  function compact(volume){
    if(!volume)return '';
    if(!Number.isFinite(volume.estimate_m3))
      return 'Unavailable · '+volume.status.replaceAll('_',' ');
    const litres=volume.estimate_m3*1000;
    const amount=litres<1?`${(litres*1000).toPrecision(3)} mL`:`${litres.toPrecision(3)} L`;
    return `~${amount} · ${volume.method==='upright_cylinder'?'cylinder estimate':'visible box proxy'}`;
  }
  function details(volume){
    if(!volume)return [];
    const lines=[compact(volume)];
    if(volume.dimensions_m)
      lines.push(`Dimensions: ${volume.dimensions_m.map(n=>(n*100).toPrecision(3)).join(' × ')} cm`);
    lines.push(`${volume.native_samples} native depth samples · physical accuracy unverified`);
    if(volume.registered_valid_fraction!=null)
      lines.push(`${(volume.registered_valid_fraction*100).toFixed(1)}% of mask has registered depth`);
    if(volume.cylinder&&volume.cylinder.status!=='conditional_estimate')
      lines.push('Cylinder estimate unavailable · '+volume.cylinder.status.replaceAll('_',' '));
    if(volume.semantics)lines.push(volume.semantics);
    for(const assumption of volume.assumptions||[])lines.push('Assumption: '+assumption);
    const core=volume.mask_sensitivity;
    if(core)lines.push(Number.isFinite(core.estimate_m3)
      ? `With mask edges removed: ${(core.estimate_m3*1000).toPrecision(3)} L (${core.method.replaceAll('_',' ')}) · sensitivity comparison`
      : `With mask edges removed: unavailable (${core.status.replaceAll('_',' ')})`);
    for(const warning of volume.warnings||[])lines.push(warning);
    return lines;
  }
  function overlay(volume){
    if(!volume)return '';
    return Number.isFinite(volume.estimate_m3)?compact(volume):'volume unavailable';
  }
  root.RGBVolume={compact,details,overlay};
})(window);
