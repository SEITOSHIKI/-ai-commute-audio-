const path = require("path");
const V = require(path.resolve(process.argv[2], "tora_v2.js"));
const { estimate, SEG, SOM } = require(path.resolve(process.argv[2], "tora_market.js"));
const out = {};
out.H = V.H; out.cv = V.customerValue(); out.unit = V.unit(); out.ltv = [3,5,7,10].map(y=>V.unit().ltv(y));
out.cogsSite = V.cogsSite(); out.marginSite = V.marginSite(); out.cloudPerDevice = V.cloudPerDevice; out.CSDAY = V.CSDAY;
out.setupCost = V.setupDays*V.CSDAY + V.setupTravel; out.leaseSite = V.leaseSite(); out.priceSite = V.priceSite();
out.team = V.team(10);
const R = V.run({years:10}); out.run10 = R;
out.run5 = (({firstBlack,cumBack,minCash,cumPeak})=>({firstBlack,cumBack,minCash,cumPeak}))(V.run());
out.margins = [5,10,25].map(n=>({n, part: V.marginSite({...V.PLAN, devices:n}, V.cloudPerDevice), full: V.marginSite({...V.PLAN, devices:n}, V.FULL_UPLOAD_PER_DEVICE)}));
out.sens = [[10,15],[15,20],[25,35]].map(([e,m])=>({e,m,...V.customerValue({electricianMin:e,managerMin:m})}));
out.sensDays = [180,200,220,240].map(d=>[[10,15],[15,20],[25,35]].map(([e,m])=>{const c=V.customerValue({electricianMin:e,managerMin:m,plan:{workDays:d}});return {d,e,m,docs:c.docs,ratio:c.ratio,be:c.breakEvenMin};}));
const d15 = { devices: 15, mix: { glass: 12, display: 3 } };
const k=0.7;
const scale10 = (k) => ({ newSites: V.GROWTH.newSites.concat(V.EXT.newSites).map(n=>Math.round(n*k)), contract: V.GROWTH.contract.concat(V.EXT.contract).map(c=>c*k) });
const SC = [
 ["計画どおり", {}],
 ["値段 −20%", { priceK: 0.8 }],
 ["解約 年20%", { churnRate: 0.2 }],
 ["AI・保存費が2倍", { perDevice: V.cloudPerDevice * 2 }],
 ["年収 +100万円／人", { salaryAdd: 100 }],
 ["新規 −30%（採用も連動）", { newSites: V.GROWTH.newSites.map(n=>Math.round(n*k)), contract: V.GROWTH.contract.map(c=>c*k), hireK: k }, { ...scale10(k), hireK: k }],
 ["新規 −30%（採用は計画どおり）", { newSites: V.GROWTH.newSites.map(n=>Math.round(n*k)), contract: V.GROWTH.contract.map(c=>c*k) }, { ...scale10(k) }],
 ["1拠点15台", { plan: d15 }],
 ["全編をクラウドで解析", { perDevice: V.FULL_UPLOAD_PER_DEVICE }],
];
const pick = (r)=>({firstBlack:r.firstBlack, cumBack:r.cumBack, op5:r.rows[4].op, minCash:r.minCash, ops:r.rows.map(x=>x.op), cash:r.rows.map(x=>x.cash)});
out.SC = SC.map(([n,o,o10])=>({n, o5: pick(V.run(o)), o10: pick(V.run({...(o10||o), years:10}))}));
// check shortfall equals V.shortfall
const sf = V.shortfall(0.7,true); out.sfCheck = [sf.rows[4].op, sf.minCash, out.SC[5].o5.op5, out.SC[5].o5.minCash];
// sales from year 3
V.ROLES[5][2] = [0,0,1,3,3];
out.salesY3 = { o5: pick(V.run()), o10: pick(V.run({years:10})) };
V.ROLES[5][2] = [0,1,1,3,3];
out.market = estimate();
// d needed in year 5
console.log(JSON.stringify(out, null, 1));
