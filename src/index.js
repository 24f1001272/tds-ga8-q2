export default {
  async fetch(request, env, ctx) {
    if (request.method !== 'POST' || new URL(request.url).pathname !== '/bqml') {
      return new Response("Not Found", { status: 404 });
    }

    let bodyText;
    try {
      bodyText = await request.text();
    } catch (e) {
      return new Response(JSON.stringify({ error: "INVALID_INPUT" }), { status: 400, headers: { 'content-type': 'application/json' } });
    }

    let data;
    try {
      data = JSON.parse(bodyText);
    } catch (e) {
      return new Response(JSON.stringify({ error: "INVALID_INPUT" }), { status: 400, headers: { 'content-type': 'application/json' } });
    }

    if (!data || typeof data !== 'object') {
      return new Response(JSON.stringify({ error: "INVALID_INPUT" }), { status: 400, headers: { 'content-type': 'application/json' } });
    }

    if (data.phase === 'select') {
      return handleSelect(data, bodyText, env);
    } else if (data.phase === 'evaluate') {
      return handleEvaluate(data, env);
    } else {
      return new Response(JSON.stringify({ error: "INVALID_INPUT" }), { status: 400, headers: { 'content-type': 'application/json' } });
    }
  }
};

function utf8Compare(a, b) {
  const enc = new TextEncoder();
  const aBytes = enc.encode(a);
  const bBytes = enc.encode(b);
  const len = Math.min(aBytes.length, bBytes.length);
  for (let i = 0; i < len; i++) {
    if (aBytes[i] !== bBytes[i]) return aBytes[i] - bBytes[i];
  }
  return aBytes.length - bBytes.length;
}

async function handleSelect(data, bodyText, env) {
  const reasonCodes = [];
  let isMalformed = false;
  const dateRegex = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,3})?(Z|[+-]\d{2}:\d{2})$/;

  // 1. Basic Type & Presence Checks
  if (!data || typeof data !== 'object') {
    isMalformed = true;
  } else {
    if (!data.runId || typeof data.runId !== 'string' || data.runId.length === 0 || data.runId.length > 128) {
      isMalformed = true;
    }
    if (data.forbiddenFeatures !== undefined && !Array.isArray(data.forbiddenFeatures)) {
      isMalformed = true;
    }
    if (typeof data.numTrialsLimit !== 'number' || !Number.isInteger(data.numTrialsLimit) || data.numTrialsLimit <= 0) {
      isMalformed = true;
    }

    if (!Array.isArray(data.rows) || data.rows.length === 0) {
      isMalformed = true;
    } else {
      const rowIds = new Set();
      for (const r of data.rows) {
        if (!r || typeof r !== 'object' || typeof r.id !== 'string' || typeof r.entity !== 'string' ||
            !dateRegex.test(r.eventTime) || !dateRegex.test(r.predictionTime) ||
            typeof r.version !== 'number' || !Number.isSafeInteger(r.version) || r.version < 0 ||
            (r.split !== 'TRAIN' && r.split !== 'EVAL') ||
            !r.features || typeof r.features !== 'object') {
          isMalformed = true;
          break;
        } else {
          if (rowIds.has(r.id)) { isMalformed = true; break; }
          rowIds.add(r.id);
          
          for (const [fname, fval] of Object.entries(r.features)) {
            if (!fval || typeof fval !== 'object' || typeof fval.value !== 'string' || typeof fval.availableAt !== 'string' || !dateRegex.test(fval.availableAt)) {
              isMalformed = true;
              break;
            }
          }
        }
      }
    }

    if (!Array.isArray(data.trials)) {
      isMalformed = true;
    } else {
      const trialIds = new Set();
      for (const t of data.trials) {
        if (!t || typeof t !== 'object' || typeof t.trialId !== 'number' || !Number.isSafeInteger(t.trialId) || t.trialId < 0) {
          isMalformed = true;
          break;
        } else {
          if (trialIds.has(t.trialId)) { isMalformed = true; break; }
          trialIds.add(t.trialId);
          if (t.status !== 'SUCCEEDED' && t.status !== 'FAILED') { isMalformed = true; break; }
        }
      }
    }
  }

  if (isMalformed) {
    reasonCodes.push('INVALID_INPUT');
  }

  // Check trial limits
  if (data && Array.isArray(data.trials) && typeof data.numTrialsLimit === 'number' && data.trials.length > data.numTrialsLimit) {
    reasonCodes.push('TRIAL_LIMIT_EXCEEDED');
  }

  // Find best successful trial
  let bestTrial = null;
  let hasSuccess = false;
  if (data && Array.isArray(data.trials)) {
    for (const trial of data.trials) {
      if (trial && trial.status === 'SUCCEEDED') {
        hasSuccess = true;
        if (typeof trial.evalMetric === 'number' && isFinite(trial.evalMetric) && typeof trial.trialId === 'number' && Number.isSafeInteger(trial.trialId) && trial.trialId >= 0) {
          if (!bestTrial) {
            bestTrial = trial;
          } else if (trial.evalMetric > bestTrial.evalMetric) {
            bestTrial = trial;
          } else if (trial.evalMetric === bestTrial.evalMetric) {
            if (trial.trialId < bestTrial.trialId) {
              bestTrial = trial;
            }
          }
        }
      }
    }
  }

  if (!hasSuccess) {
    reasonCodes.push('NO_SUCCESSFUL_TRIAL');
  }

  let selectedTrialId = null;
  let datasetDigest = null;
  let trainRowIds = [];
  let evalRowIds = [];
  let featureNames = [];

  // Deduplicate and process rows only if not malformed
  if (!reasonCodes.includes('INVALID_INPUT')) {
    const retainedRowsMap = new Map();
    for (const row of data.rows) {
      const eventTimeNum = new Date(row.eventTime).getTime();
      const key = `${row.entity}|${eventTimeNum}`;
      if (retainedRowsMap.has(key)) {
        const existing = retainedRowsMap.get(key);
        if (row.version > existing.version) {
          retainedRowsMap.set(key, row);
        } else if (row.version === existing.version) {
          if (utf8Compare(row.id, existing.id) < 0) {
            retainedRowsMap.set(key, row);
          }
        }
      } else {
        retainedRowsMap.set(key, row);
      }
    }

    const retainedRows = Array.from(retainedRowsMap.values());
    
    // Feature eligibility
    if (retainedRows.length > 0) {
      const firstRowFeatures = Object.keys(retainedRows[0].features || {});
      const forbidden = data.forbiddenFeatures || [];
      
      for (const f of firstRowFeatures) {
        if (forbidden.includes(f)) continue;
        
        let eligible = true;
        for (const row of retainedRows) {
          if (!row.features[f]) {
            eligible = false;
            break;
          }
          const availDate = new Date(row.features[f].availableAt);
          const predDate = new Date(row.predictionTime);
          if (availDate.getTime() > predDate.getTime()) {
            eligible = false;
            break;
          }
        }
        if (eligible) {
          featureNames.push(f);
        }
      }
      
      for (const row of retainedRows) {
        if (row.split === 'TRAIN') trainRowIds.push(row.id);
        if (row.split === 'EVAL') evalRowIds.push(row.id);
      }
      
      featureNames.sort(utf8Compare);
      trainRowIds.sort(utf8Compare);
      evalRowIds.sort(utf8Compare);

      const digestObj = {
        trainRowIds: trainRowIds,
        evalRowIds: evalRowIds,
        featureNames: featureNames
      };
      
      const compactJson = JSON.stringify(digestObj);
      const msgBuffer = new TextEncoder().encode(compactJson);
      const hashBuffer = await crypto.subtle.digest('SHA-256', msgBuffer);
      const hashArray = Array.from(new Uint8Array(hashBuffer));
      datasetDigest = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
      
      if (bestTrial) {
        selectedTrialId = bestTrial.trialId;
      }
    }
  }

  reasonCodes.sort(utf8Compare);

  const responseObj = {
    runId: data && data.runId ? data.runId : null,
    selectedTrialId: reasonCodes.length > 0 ? null : selectedTrialId,
    trainRowIds: reasonCodes.includes('INVALID_INPUT') ? [] : trainRowIds,
    evalRowIds: reasonCodes.includes('INVALID_INPUT') ? [] : evalRowIds,
    featureNames: reasonCodes.includes('INVALID_INPUT') ? [] : featureNames,
    datasetDigest: reasonCodes.includes('INVALID_INPUT') ? null : datasetDigest,
    reasonCodes: [...new Set(reasonCodes)]
  };

  // Only store if runId was somewhat valid
  if (data && data.runId && typeof data.runId === 'string' && data.runId.length > 0) {
     const storedStr = await env.STORE.get(data.runId);
     if (storedStr) {
       const parsed = JSON.parse(storedStr);
       if (parsed.request === bodyText) {
         return new Response(JSON.stringify(parsed.response), { status: 200, headers: { 'content-type': 'application/json' } });
       } else {
         return new Response(JSON.stringify({ error: "RUN_ID_CONFLICT" }), { status: 409, headers: { 'content-type': 'application/json' } });
       }
     }
     await env.STORE.put(data.runId, JSON.stringify({ request: bodyText, response: responseObj }));
  }

  return new Response(JSON.stringify(responseObj), { status: 200, headers: { 'content-type': 'application/json' } });
}

async function handleEvaluate(data, env) {
  let reasonCodes = [];
  let isMalformed = false;

  if (!data || typeof data !== 'object') {
    isMalformed = true;
  } else {
    if (!data.runId || typeof data.runId !== 'string' || !data.datasetDigest || typeof data.datasetDigest !== 'string' || typeof data.selectedTrialId !== 'number' || !Number.isSafeInteger(data.selectedTrialId)) {
      isMalformed = true;
    }
    if (typeof data.bytesProcessed !== 'number' || !Number.isSafeInteger(data.bytesProcessed) || data.bytesProcessed < 0) isMalformed = true;
    if (typeof data.maxBytes !== 'number' || !Number.isSafeInteger(data.maxBytes) || data.maxBytes < 0) isMalformed = true;
    if (typeof data.metricFloor !== 'number' || data.metricFloor < 0 || data.metricFloor > 1 || !isFinite(data.metricFloor)) isMalformed = true;
    
    if (data.requiredSlices !== undefined && data.requiredSlices !== null) {
      if (typeof data.requiredSlices !== 'object') {
        isMalformed = true;
      } else {
        for (const [k, v] of Object.entries(data.requiredSlices)) {
          if (typeof v !== 'number' || v < 0 || v > 1 || !isFinite(v)) {
            isMalformed = true;
            break;
          }
        }
      }
    }
  }

  if (isMalformed) {
    reasonCodes.push('INVALID_INPUT');
  }

  if (data && typeof data.bytesProcessed === 'number' && typeof data.maxBytes === 'number' && data.bytesProcessed > data.maxBytes) {
    reasonCodes.push('BYTE_LIMIT');
  }

  let invalidLineage = false;
  if (data && typeof data.runId === 'string' && typeof data.datasetDigest === 'string' && typeof data.selectedTrialId === 'number') {
    const storedStr = await env.STORE.get(data.runId);
    if (!storedStr) {
      invalidLineage = true;
    } else {
      const stored = JSON.parse(storedStr);
      const sel = stored.response;
      if (sel.selectedTrialId !== data.selectedTrialId || sel.datasetDigest !== data.datasetDigest || sel.selectedTrialId === null) {
        invalidLineage = true;
      }
    }
  } else {
    invalidLineage = true;
  }

  if (invalidLineage) {
    reasonCodes.push('INVALID_LINEAGE');
  }

  let hasInvalidRow = false;
  let rowCount = 0;
  let correctCount = 0;
  let sliceCorrect = {};
  let sliceTotal = {};
  let presentSlices = new Set();
  let rowsEmpty = false;

  if (!data || !Array.isArray(data.rows)) {
    if (!reasonCodes.includes('INVALID_INPUT')) reasonCodes.push('INVALID_INPUT');
    rowsEmpty = true;
  } else {
    if (data.rows.length === 0) rowsEmpty = true;
    for (const row of data.rows) {
      if (!row || typeof row !== 'object') { hasInvalidRow = true; break; }
      if (row.label !== 0 && row.label !== 1) { hasInvalidRow = true; break; }
      if (row.prediction !== 0 && row.prediction !== 1) { hasInvalidRow = true; break; }
      if (typeof row.slice !== 'string' || row.slice === '') { hasInvalidRow = true; break; }

      rowCount++;
      if (row.label === row.prediction) correctCount++;
      
      presentSlices.add(row.slice);
      sliceTotal[row.slice] = (sliceTotal[row.slice] || 0) + 1;
      if (row.label === row.prediction) {
        sliceCorrect[row.slice] = (sliceCorrect[row.slice] || 0) + 1;
      }
    }
  }

  if (hasInvalidRow) {
    reasonCodes.push('INVALID_TEST_ROW');
  }

  let testMetric = null;
  let criticalSlicePass = true;

  if (isMalformed || invalidLineage || hasInvalidRow) {
    criticalSlicePass = false;
  }

  if (hasInvalidRow || rowsEmpty || isMalformed || reasonCodes.includes('INVALID_INPUT')) {
    testMetric = null;
  } else {
    testMetric = correctCount / rowCount;
    testMetric = Number(testMetric.toFixed(12));
    
    if (testMetric < data.metricFloor) {
      reasonCodes.push('AGGREGATE_FLOOR');
    }
    
    if (data.requiredSlices && typeof data.requiredSlices === 'object') {
      for (const [sliceName, floor] of Object.entries(data.requiredSlices)) {
        if (!presentSlices.has(sliceName)) {
          reasonCodes.push(`MISSING_SLICE:${sliceName}`);
          criticalSlicePass = false;
        } else {
          let sliceAcc = sliceCorrect[sliceName] / sliceTotal[sliceName];
          sliceAcc = Number(sliceAcc.toFixed(12));
          if (sliceAcc < floor) {
            reasonCodes.push(`SLICE_FLOOR:${sliceName}`);
            criticalSlicePass = false;
          }
        }
      }
    }
  }

  let decision = "reject";
  if (reasonCodes.length === 0) {
    decision = "admit";
  }

  reasonCodes = [...new Set(reasonCodes)];
  reasonCodes.sort(utf8Compare);

  const responseObj = {
    runId: data && data.runId ? data.runId : null,
    selectedTrialId: data && typeof data.selectedTrialId === 'number' ? data.selectedTrialId : null,
    datasetDigest: data && typeof data.datasetDigest === 'string' ? data.datasetDigest : null,
    testMetric: testMetric,
    criticalSlicePass: criticalSlicePass,
    decision: decision,
    bytesProcessed: data && typeof data.bytesProcessed === 'number' ? data.bytesProcessed : null,
    reasonCodes: reasonCodes
  };

  return new Response(JSON.stringify(responseObj), { status: 200, headers: { 'content-type': 'application/json' } });
}

