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
  
  if (!data.runId || typeof data.runId !== 'string' || data.runId.length === 0 || data.runId.length > 128) {
    return new Response(JSON.stringify({ error: "INVALID_INPUT" }), { status: 400, headers: { 'content-type': 'application/json' } });
  }

  const storedStr = await env.STORE.get(data.runId);
  if (storedStr) {
    const parsed = JSON.parse(storedStr);
    if (parsed.request === bodyText) {
      return new Response(JSON.stringify(parsed.response), { status: 200, headers: { 'content-type': 'application/json' } });
    } else {
      return new Response(JSON.stringify({ error: "RUN_ID_CONFLICT" }), { status: 409, headers: { 'content-type': 'application/json' } });
    }
  }

  let isMalformed = false;
  if (!Array.isArray(data.rows) || !Array.isArray(data.trials) || !Number.isInteger(data.numTrialsLimit) || data.numTrialsLimit <= 0) {
    isMalformed = true;
    reasonCodes.push('INVALID_INPUT');
  }

  let selectedTrialId = null;
  let datasetDigest = null;
  let trainRowIds = [];
  let evalRowIds = [];
  let featureNames = [];

  if (!isMalformed) {
    if (data.trials.length > data.numTrialsLimit) {
      reasonCodes.push('TRIAL_LIMIT_EXCEEDED');
    }

    let bestTrial = null;
    let hasSuccess = false;
    for (const trial of data.trials) {
      if (trial.status === 'SUCCEEDED') {
        hasSuccess = true;
        if (typeof trial.evalMetric === 'number' && isFinite(trial.evalMetric) && Number.isSafeInteger(trial.trialId) && trial.trialId >= 0) {
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
    
    if (!hasSuccess) {
      reasonCodes.push('NO_SUCCESSFUL_TRIAL');
    }

    const retainedRowsMap = new Map();
    let rowMalformed = false;
    
    const dateRegex = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,3})?(Z|[+-]\d{2}:\d{2})$/;

    for (const row of data.rows) {
      if (!row.id || !row.entity || !row.eventTime || !row.predictionTime || !Number.isSafeInteger(row.version) || row.version < 0 || !['TRAIN', 'EVAL'].includes(row.split) || !row.features) {
        rowMalformed = true;
        continue;
      }
      if (!dateRegex.test(row.eventTime) || !dateRegex.test(row.predictionTime)) {
        rowMalformed = true;
        continue;
      }
      
      const eventTimeNum = new Date(row.eventTime).getTime();
      if (isNaN(eventTimeNum)) {
        rowMalformed = true;
        continue;
      }

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
    
    if (rowMalformed && !reasonCodes.includes('INVALID_INPUT')) {
      reasonCodes.push('INVALID_INPUT');
    }
    
    const retainedRows = Array.from(retainedRowsMap.values());
    if (retainedRows.length === 0 && !reasonCodes.includes('INVALID_INPUT')) {
      reasonCodes.push('INVALID_INPUT');
    }

    if (!reasonCodes.includes('INVALID_INPUT')) {
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
          if (isNaN(availDate.getTime()) || isNaN(predDate.getTime()) || availDate.getTime() > predDate.getTime()) {
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
    runId: data.runId,
    selectedTrialId: reasonCodes.length > 0 ? null : selectedTrialId,
    trainRowIds: reasonCodes.includes('INVALID_INPUT') ? [] : trainRowIds,
    evalRowIds: reasonCodes.includes('INVALID_INPUT') ? [] : evalRowIds,
    featureNames: reasonCodes.includes('INVALID_INPUT') ? [] : featureNames,
    datasetDigest: reasonCodes.includes('INVALID_INPUT') ? null : datasetDigest,
    reasonCodes: [...new Set(reasonCodes)]
  };

  await env.STORE.put(data.runId, JSON.stringify({ request: bodyText, response: responseObj }));
  
  return new Response(JSON.stringify(responseObj), { status: 200, headers: { 'content-type': 'application/json' } });
}

async function handleEvaluate(data, env) {
  let reasonCodes = [];
  let isMalformed = false;

  if (!data.runId || !data.datasetDigest || typeof data.selectedTrialId !== 'number') {
    isMalformed = true;
    reasonCodes.push('INVALID_INPUT');
  }

  let invalidLineage = false;
  if (!isMalformed) {
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
    if (invalidLineage) {
      reasonCodes.push('INVALID_LINEAGE');
    }
  }

  if (typeof data.bytesProcessed !== 'number' || !Number.isSafeInteger(data.bytesProcessed) || data.bytesProcessed < 0 ||
      typeof data.maxBytes !== 'number' || !Number.isSafeInteger(data.maxBytes) || data.maxBytes < 0) {
    if (!reasonCodes.includes('INVALID_INPUT')) reasonCodes.push('INVALID_INPUT');
    isMalformed = true;
  } else {
    if (data.bytesProcessed > data.maxBytes) {
      reasonCodes.push('BYTE_LIMIT');
    }
  }

  if (!data.rows || !Array.isArray(data.rows) || typeof data.metricFloor !== 'number' || data.metricFloor < 0 || data.metricFloor > 1) {
    if (!reasonCodes.includes('INVALID_INPUT')) reasonCodes.push('INVALID_INPUT');
    isMalformed = true;
  }

  let hasInvalidRow = false;
  let rowCount = 0;
  let correctCount = 0;
  let sliceCorrect = {};
  let sliceTotal = {};
  let presentSlices = new Set();
  
  if (Array.isArray(data.rows)) {
    for (const row of data.rows) {
      if (row.label !== 0 && row.label !== 1) hasInvalidRow = true;
      if (row.prediction !== 0 && row.prediction !== 1) hasInvalidRow = true;
      if (typeof row.slice !== 'string' || row.slice === '') hasInvalidRow = true;
      
      if (hasInvalidRow) {
        if (!reasonCodes.includes('INVALID_TEST_ROW')) {
          reasonCodes.push('INVALID_TEST_ROW'); 
        }
        break; 
      }
      
      rowCount++;
      if (row.label === row.prediction) correctCount++;
      
      presentSlices.add(row.slice);
      sliceTotal[row.slice] = (sliceTotal[row.slice] || 0) + 1;
      if (row.label === row.prediction) {
        sliceCorrect[row.slice] = (sliceCorrect[row.slice] || 0) + 1;
      }
    }
  } else {
    hasInvalidRow = true;
  }

  let testMetric = null;
  let criticalSlicePass = true;
  let rowsEmpty = Array.isArray(data.rows) && data.rows.length === 0;

  if (isMalformed || invalidLineage) {
    criticalSlicePass = false;
  }
  if (hasInvalidRow) {
    criticalSlicePass = false;
  }

  if (hasInvalidRow || rowsEmpty || isMalformed) {
    testMetric = null;
  } else {
    testMetric = correctCount / rowCount;
    testMetric = Math.round(testMetric * 1e12) / 1e12;
    
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
          sliceAcc = Math.round(sliceAcc * 1e12) / 1e12;
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
    runId: data.runId,
    selectedTrialId: data.selectedTrialId,
    datasetDigest: data.datasetDigest,
    testMetric: testMetric,
    criticalSlicePass: criticalSlicePass,
    decision: decision,
    bytesProcessed: data.bytesProcessed,
    reasonCodes: reasonCodes
  };

  return new Response(JSON.stringify(responseObj), { status: 200, headers: { 'content-type': 'application/json' } });
}
