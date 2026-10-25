"""Select the displayed value without changing recorded model readings."""


def value(initial, candidate, settings):
    if settings.get('display_mode', 'objects') == 'materials':
        readings = [m for m in (candidate or {}).get('materials', []) if m['confidence'] >= settings['vlm_threshold']]
        empty = ('No material above threshold' if 'materials' in (candidate or {}) else 'Materials not read') if candidate else 'Materials pending'
        return {'label': ' + '.join(m['label'] for m in readings) or empty,
                'confidence': min((m['confidence'] for m in readings), default=None),
                'provider': 'vlm' if readings else 'pending'}
    accepted = bool(candidate and candidate['confidence'] >= settings['vlm_threshold'])
    return {'label': candidate['label'] if accepted else initial['label'],
            'confidence': candidate['confidence'] if accepted else initial['confidence'],
            'provider': 'vlm' if accepted else 'detector'}
