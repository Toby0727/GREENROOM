const crypto = require('crypto');

module.exports = async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).end();

  const APP_ID  = '2128688';
  const KEY     = '8e1cefae4252719fab23';
  const SECRET  = '17bdb8a6f3c6cb79486d';
  const CLUSTER = 'us2';

  const { event, data } = req.body;
  const body = JSON.stringify({ name: event, channel: 'livedoc', data: JSON.stringify(data) });
  const bodyMd5 = crypto.createHash('md5').update(body).digest('hex');
  const timestamp = Math.floor(Date.now() / 1000);

  const params = [
    `auth_key=${KEY}`,
    `auth_timestamp=${timestamp}`,
    `auth_version=1.0`,
    `body_md5=${bodyMd5}`
  ].sort().join('&');

  const stringToSign = `POST\n/apps/${APP_ID}/events\n${params}`;
  const signature = crypto.createHmac('sha256', SECRET).update(stringToSign).digest('hex');

  const url = `https://api-${CLUSTER}.pusher.com/apps/${APP_ID}/events?${params}&auth_signature=${signature}`;

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body
    });
    const text = await response.text();
    res.status(response.status).send(text);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
}
